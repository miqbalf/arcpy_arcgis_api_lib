"""Independent web-map cloning with Feature Service data duplication.

Two-phase flow (per environment suffix):
1. Clone hosted Feature Service(s) with ``copy_data=True`` while polling layer
   feature counts (heartbeat + stall timeout + JSON status file).
2. Clone the Web Map with ``item_mapping`` so the map points at the new
   service(s) without re-copying data.

This avoids the opaque all-in-one ``clone_items([webmap])`` hang where progress
is invisible for long periods.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from arcgis.gis import GIS, Item

logger = logging.getLogger("clone_webmap")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _walk_operational_layers(webmap_data: dict):
    for layer_group in webmap_data.get("operationalLayers", []) or []:
        yield layer_group
        for child in layer_group.get("layers", []) or []:
            yield child


def extract_service_urls(webmap_data: dict) -> Set[str]:
    urls: Set[str] = set()
    for layer in _walk_operational_layers(webmap_data):
        url = layer.get("url")
        if url:
            urls.add(url.split("?")[0].rstrip("/"))
    return urls


def extract_item_ids(webmap_data: dict) -> Set[str]:
    ids: Set[str] = set()
    for layer in _walk_operational_layers(webmap_data):
        if layer.get("itemId"):
            ids.add(layer["itemId"])
    return ids


def get_webmap_item(gis: GIS, title: str) -> Item:
    matches = [
        it
        for it in gis.content.search(
            query=f'title:"{title}" AND type:"Web Map"',
            max_items=20,
        )
        if it.title == title
    ]
    if not matches:
        raise ValueError(f"Web Map not found: {title!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple Web Maps found with title {title!r}")
    return matches[0]


def layer_feature_counts(fs_item: Item) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lyr in fs_item.layers or []:
        props = lyr.properties
        try:
            count = lyr.query(return_count_only=True, where="1=1")
        except Exception as exc:  # noqa: BLE001 - surface in status file
            count = None
            err = str(exc)
        else:
            err = None
        rows.append(
            {
                "id": int(props.id),
                "name": props.name,
                "count": count,
                "error": err,
            }
        )
    return rows


def summarize_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    filled = sum(1 for r in rows if (r.get("count") or 0) > 0)
    empty = [r for r in rows if not (r.get("count") or 0)]
    errored = [r for r in rows if r.get("error")]
    return {
        "filled": filled,
        "total": len(rows),
        "empty_layer_ids": [r["id"] for r in empty],
        "empty_layer_names": [r["name"] for r in empty],
        "error_layers": [{"id": r["id"], "name": r["name"], "error": r["error"]} for r in errored],
        "counts": {str(r["id"]): r.get("count") for r in rows},
    }


def counts_match(source_rows: List[Dict[str, Any]], clone_rows: List[Dict[str, Any]]) -> bool:
    if len(source_rows) != len(clone_rows):
        return False
    src = {r["id"]: r.get("count") for r in source_rows}
    dst = {r["id"]: r.get("count") for r in clone_rows}
    return src == dst


class CloneValidationError(RuntimeError):
    """Raised when source/clone data looks wrong, incomplete, or unsafe to continue."""


def validate_feature_service_rows(
    label: str,
    rows: List[Dict[str, Any]],
    *,
    require_non_empty: bool = True,
    min_layers: int = 1,
) -> Dict[str, Any]:
    """Fail fast on unreadable counts, empty services, or other weird layer stats."""
    if len(rows) < min_layers:
        raise CloneValidationError(f"{label}: expected at least {min_layers} layer(s), got {len(rows)}")

    errored = [r for r in rows if r.get("error") is not None or r.get("count") is None]
    if errored:
        details = ", ".join(f"L{r['id']} {r['name']}: {r.get('error') or 'count=None'}" for r in errored)
        raise CloneValidationError(f"{label}: cannot read feature counts ({details})")

    if require_non_empty:
        empty = [r for r in rows if (r.get("count") or 0) <= 0]
        if empty:
            names = [f"L{r['id']} {r['name']}" for r in empty]
            raise CloneValidationError(f"{label}: unexpected empty layer(s): {names}")

    names = [r["name"] for r in rows]
    if len(names) != len(set(names)):
        raise CloneValidationError(f"{label}: duplicate layer names detected: {names}")

    return summarize_counts(rows)


def assert_clone_matches_source(
    source_rows: List[Dict[str, Any]],
    clone_rows: List[Dict[str, Any]],
    *,
    label: str = "clone FS",
) -> None:
    """Require exact layer id/name/count match with the source Feature Service."""
    src_by_id = {r["id"]: r for r in source_rows}
    dst_by_id = {r["id"]: r for r in clone_rows}

    if set(src_by_id) != set(dst_by_id):
        raise CloneValidationError(
            f"{label}: layer id mismatch source={sorted(src_by_id)} clone={sorted(dst_by_id)}"
        )

    problems: List[str] = []
    for lid in sorted(src_by_id):
        s, d = src_by_id[lid], dst_by_id[lid]
        if s.get("name") != d.get("name"):
            problems.append(f"L{lid} name {s.get('name')!r} != {d.get('name')!r}")
        if s.get("count") != d.get("count"):
            problems.append(f"L{lid} {s.get('name')}: count {d.get('count')} != source {s.get('count')}")
        if d.get("error"):
            problems.append(f"L{lid} {d.get('name')}: {d.get('error')}")
    if problems:
        raise CloneValidationError(f"{label}: data mismatch / weird dataset: {problems}")


@dataclass
class CloneStatusWriter:
    path: Path
    payload: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> None:
        self.payload.update(kwargs)
        self.payload["updated_at"] = _utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")
        logger.info("status -> %s", self.path)


class FeatureServiceCloneMonitor:
    """Poll feature counts while a clone job runs in another thread."""

    def __init__(
        self,
        gis: GIS,
        source_fs: Item,
        folder: Optional[str],
        status: CloneStatusWriter,
        heartbeat_sec: int = 30,
        stall_timeout_sec: int = 600,
    ):
        self.gis = gis
        self.source_fs = source_fs
        self.folder = folder
        self.status = status
        self.heartbeat_sec = heartbeat_sec
        self.stall_timeout_sec = stall_timeout_sec
        self.source_counts = layer_feature_counts(source_fs)
        self.stop = threading.Event()
        self.stall_detected = threading.Event()
        self.latest_clone_id: Optional[str] = None
        self._last_signature: Optional[str] = None
        self._last_progress_at = time.time()

    def _find_candidate(self) -> Optional[Item]:
        """Find the newest Feature Service clone of the source in the target folder.

        Only exact title ``SOURCE`` or ``SOURCE_<suffix>`` are accepted. Avoids
        ArcGIS fuzzy title search matching unrelated items like
        ``Gara point for clusters`` when source is ``GAR``.
        """
        allowed_titles = {
            self.source_fs.title,
            f"{self.source_fs.title}_stg",
            f"{self.source_fs.title}_dev",
        }
        try:
            items = list_folder_items(self.gis, self.folder)
        except Exception:  # noqa: BLE001
            items = []
        candidates = [
            it
            for it in items
            if it.type == "Feature Service"
            and it.id != self.source_fs.id
            and it.title in allowed_titles
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda it: it.created, reverse=True)
        return candidates[0]

    def run(self) -> None:
        folder_label = self.folder or "(root)"
        self.status.update(
            phase="clone_feature_service",
            source_fs_id=self.source_fs.id,
            source_counts=summarize_counts(self.source_counts),
            message=f"waiting for clone Feature Service to appear in {folder_label}",
        )
        while not self.stop.wait(self.heartbeat_sec):
            candidate = self._find_candidate()
            if not candidate:
                idle = time.time() - self._last_progress_at
                self.status.update(
                    message=f"no clone Feature Service in {folder_label} yet",
                    idle_sec=round(idle, 1),
                )
                if idle >= self.stall_timeout_sec:
                    self.stall_detected.set()
                    self.status.update(
                        state="STALLED",
                        error=f"No Feature Service appeared in {folder_label} within {self.stall_timeout_sec}s",
                    )
                    return
                continue

            self.latest_clone_id = candidate.id
            rows = layer_feature_counts(candidate)
            # Guard against latching onto an unrelated item
            if len(rows) != len(self.source_counts):
                idle = time.time() - self._last_progress_at
                self.status.update(
                    message=(
                        f"ignoring candidate {candidate.title!r} ({candidate.id}): "
                        f"{len(rows)} layers vs source {len(self.source_counts)}"
                    ),
                    idle_sec=round(idle, 1),
                    weird_candidate={"id": candidate.id, "title": candidate.title, "layers": len(rows)},
                )
                if idle >= self.stall_timeout_sec:
                    self.stall_detected.set()
                    self.status.update(
                        state="STALLED",
                        error=(
                            f"No valid Feature Service clone of {self.source_fs.title!r} "
                            f"with {len(self.source_counts)} layers found within {self.stall_timeout_sec}s "
                            f"(last weird candidate: {candidate.title!r})"
                        ),
                    )
                    return
                continue

            summary = summarize_counts(rows)
            signature = json.dumps(summary["counts"], sort_keys=True)
            if signature != self._last_signature:
                self._last_signature = signature
                self._last_progress_at = time.time()
                logger.info(
                    "FS clone progress %s: filled %s/%s empty=%s",
                    candidate.id,
                    summary["filled"],
                    summary["total"],
                    summary["empty_layer_names"],
                )
            idle = time.time() - self._last_progress_at
            complete = counts_match(self.source_counts, rows)
            self.status.update(
                clone_fs_id=candidate.id,
                clone_fs_title=candidate.title,
                clone_counts=summary,
                idle_sec=round(idle, 1),
                complete=complete,
                message="layer copy in progress" if not complete else "feature counts match source",
                state="COMPLETE" if complete else "RUNNING",
            )
            if complete:
                return
            if idle >= self.stall_timeout_sec:
                self.stall_detected.set()
                self.status.update(
                    state="STALLED",
                    error=(
                        f"No feature-count change for {self.stall_timeout_sec}s "
                        f"(filled {summary['filled']}/{summary['total']}; "
                        f"empty={summary['empty_layer_names']})"
                    ),
                )
                return


def item_folder_title(gis: GIS, item: Item) -> Optional[str]:
    """Return the portal folder title for an item, or None for the user root folder."""
    folder_id = getattr(item, "ownerFolder", None)
    if not folder_id:
        return None
    me = gis.users.me
    for f in me.folders:
        fid = f["id"] if isinstance(f, dict) else getattr(f, "id", None)
        title = f["title"] if isinstance(f, dict) else getattr(f, "title", None)
        if fid == folder_id:
            return title
    return None


def list_folder_items(gis: GIS, folder: Optional[str], max_items: int = 1000) -> List[Item]:
    """List items in a user folder. ``folder=None`` means root."""
    me = gis.users.me
    return list(me.items(folder=folder, max_items=max_items))


def list_incomplete_fs_clones(
    gis: GIS,
    source_fs: Item,
    folder: Optional[str] = None,
    title_suffixes: Optional[List[str]] = None,
) -> List[Tuple[Item, Dict[str, Any]]]:
    """Find incomplete Feature Service clones in the same folder as the source.

    Matches items whose title equals the source title or ``source_title + suffix``,
    excluding the source item itself, when layer counts do not match the source.
    """
    if folder is None:
        folder = item_folder_title(gis, source_fs)
    source_rows = layer_feature_counts(source_fs)
    suffixes = title_suffixes or ["_stg", "_dev"]
    allowed_titles = {source_fs.title, *(f"{source_fs.title}{s}" for s in suffixes)}
    out: List[Tuple[Item, Dict[str, Any]]] = []
    for it in list_folder_items(gis, folder):
        if it.type != "Feature Service" or it.id == source_fs.id:
            continue
        if it.title not in allowed_titles and not it.title.startswith(source_fs.title):
            continue
        rows = layer_feature_counts(it)
        summary = summarize_counts(rows)
        if not counts_match(source_rows, rows):
            out.append((it, summary))
    return out


def delete_items(items: Iterable[Item], dry_run: bool = True) -> List[Dict[str, str]]:
    results = []
    for it in items:
        entry = {"title": it.title, "type": it.type, "id": it.id}
        if dry_run:
            entry["action"] = "would_delete"
        else:
            ok = it.delete(force=True)
            entry["action"] = "deleted" if ok else "delete_failed"
        results.append(entry)
        logger.info("%s %s (%s)", entry["action"], entry["title"], entry["id"])
    return results


def rename_with_suffix(item: Item, suffix: str, source_title: Optional[str] = None) -> Item:
    if item.title.endswith(suffix):
        return item
    if source_title and item.title == source_title:
        new_title = f"{source_title}{suffix}"
    else:
        new_title = f"{item.title}{suffix}"
    item.update({"title": new_title})
    return item


def clone_feature_service_monitored(
    gis: GIS,
    source_fs: Item,
    folder: Optional[str],
    status: CloneStatusWriter,
    heartbeat_sec: int = 30,
    stall_timeout_sec: int = 600,
    overall_timeout_sec: int = 10_800,
) -> Item:
    """Clone a hosted Feature Service into ``folder`` (None = root) with progress monitoring."""
    monitor = FeatureServiceCloneMonitor(
        gis=gis,
        source_fs=source_fs,
        folder=folder,
        status=status,
        heartbeat_sec=heartbeat_sec,
        stall_timeout_sec=stall_timeout_sec,
    )

    def _clone():
        kwargs = dict(
            items=[source_fs],
            copy_data=True,
            search_existing_items=False,
            use_org_basemap=False,
        )
        if folder is not None:
            kwargs["folder"] = folder
        return gis.content.clone_items(**kwargs)

    status.update(
        state="RUNNING",
        phase="clone_feature_service",
        started_at=_utc_now(),
        folder=folder or "(root)",
        source_fs_id=source_fs.id,
        source_fs_title=source_fs.title,
    )

    monitor_thread = threading.Thread(target=monitor.run, name="fs-clone-monitor", daemon=True)
    monitor_thread.start()

    err: Optional[BaseException] = None
    cloned: List[Item] = []
    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_clone)
        try:
            while True:
                try:
                    cloned = fut.result(timeout=heartbeat_sec)
                    break
                except FuturesTimeout:
                    if monitor.stall_detected.is_set():
                        err = TimeoutError(status.payload.get("error") or "clone stalled")
                        break
                    if time.time() - wall_start >= overall_timeout_sec:
                        err = TimeoutError(
                            f"Overall timeout {overall_timeout_sec}s exceeded while cloning Feature Service"
                        )
                        break
        except Exception as exc:  # noqa: BLE001
            err = exc

    monitor.stop.set()
    monitor_thread.join(timeout=heartbeat_sec + 5)

    if err:
        status.update(state="FAILED", error=str(err), finished_at=_utc_now())
        raise RuntimeError(str(err)) from err

    fs_items = [it for it in cloned if it.type == "Feature Service"]
    if not fs_items and monitor.latest_clone_id:
        fs_items = [gis.content.get(monitor.latest_clone_id)]
    if not fs_items:
        status.update(state="FAILED", error="clone_items returned no Feature Service", finished_at=_utc_now())
        raise RuntimeError("clone_items returned no Feature Service")

    fs_item = fs_items[0]
    final_rows = layer_feature_counts(fs_item)
    try:
        assert_clone_matches_source(monitor.source_counts, final_rows, label=f"FS {fs_item.id}")
    except CloneValidationError as exc:
        summary = summarize_counts(final_rows)
        status.update(
            state="FAILED",
            error=str(exc),
            clone_fs_id=fs_item.id,
            clone_counts=summary,
            finished_at=_utc_now(),
        )
        raise

    status.update(
        state="FS_OK",
        clone_fs_id=fs_item.id,
        clone_fs_title=fs_item.title,
        clone_counts=summarize_counts(final_rows),
        message="Feature Service data verified against source",
        finished_fs_at=_utc_now(),
    )
    return fs_item


def clone_webmap_with_mapping(
    gis: GIS,
    source_webmap: Item,
    folder: Optional[str],
    item_mapping: Dict[str, str],
    suffix: str,
) -> Tuple[Item, List[Item]]:
    kwargs = dict(
        items=[source_webmap],
        copy_data=False,  # data already duplicated via mapped FS
        search_existing_items=False,
        item_mapping=item_mapping,
        use_org_basemap=False,
    )
    if folder is not None:
        kwargs["folder"] = folder
    cloned = gis.content.clone_items(**kwargs)
    if not cloned:
        raise RuntimeError("Web Map clone returned no items")

    renamed: List[Item] = []
    for item in cloned:
        if item.type == "Web Map":
            rename_with_suffix(item, suffix, source_title=source_webmap.title)
        elif not item.title.endswith(suffix):
            rename_with_suffix(item, suffix)
        renamed.append(item)

    webmaps = [it for it in renamed if it.type == "Web Map"]
    if not webmaps:
        raise RuntimeError("No Web Map in clone result")
    target_title = f"{source_webmap.title}{suffix}"
    exact = [it for it in webmaps if it.title == target_title]
    return (exact[0] if exact else webmaps[0]), renamed


def verify_independent_clone(
    source_webmap: Item,
    cloned_webmap: Item,
    cloned_fs: Item,
) -> Dict[str, Any]:
    source_data = source_webmap.get_data()
    cloned_data = cloned_webmap.get_data()
    source_urls = extract_service_urls(source_data)
    cloned_urls = extract_service_urls(cloned_data)
    shared_urls = source_urls & cloned_urls
    source_ids = {source_webmap.id, *extract_item_ids(source_data)}
    overlap_ids = source_ids & {cloned_webmap.id, cloned_fs.id}
    return {
        "clone_title": cloned_webmap.title,
        "clone_id": cloned_webmap.id,
        "clone_fs_id": cloned_fs.id,
        "shared_service_urls": sorted(shared_urls),
        "shared_item_ids": sorted(overlap_ids),
        "independent_data": not shared_urls and not overlap_ids,
        "clone_service_urls": sorted(cloned_urls),
    }


def assert_independent_webmap(
    source_webmap: Item,
    cloned_webmap: Item,
    cloned_fs: Item,
) -> Dict[str, Any]:
    report = verify_independent_clone(source_webmap, cloned_webmap, cloned_fs)
    if report["shared_service_urls"]:
        raise CloneValidationError(
            f"Web map {cloned_webmap.title!r} still references source service URL(s): "
            f"{report['shared_service_urls']}"
        )
    if report["shared_item_ids"]:
        raise CloneValidationError(
            f"Web map {cloned_webmap.title!r} still shares item id(s) with source: "
            f"{report['shared_item_ids']}"
        )
    if not report["independent_data"]:
        raise CloneValidationError(f"Web map {cloned_webmap.title!r} is not an independent data clone")
    return report


def clone_webmap_environment(
    gis: GIS,
    source_webmap: Item,
    suffix: str,
    folder: Optional[str] = None,
    status_dir: Path | str = "clone_status",
    heartbeat_sec: int = 30,
    stall_timeout_sec: int = 600,
    skip_if_webmap_exists: bool = True,
    require_clean: bool = True,
    expected_fs_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """Clone one environment (e.g. ``_stg``) into the same folder as the source.

    Only item titles are suffixed (e.g. ``Catalyze design_GARA_project_data_dev``).
    Stops immediately on validation errors, incomplete data, or non-independent maps.
    """
    if folder is None:
        folder = item_folder_title(gis, source_webmap)
    target_title = f"{source_webmap.title}{suffix}"
    status_path = Path(status_dir) / f"clone{suffix}.json"
    status = CloneStatusWriter(
        status_path,
        {"suffix": suffix, "folder": folder or "(root)", "target_title": target_title},
    )

    existing = [
        it
        for it in gis.content.search(
            query=f'title:"{target_title}" AND type:"Web Map"',
            max_items=10,
        )
        if it.title == target_title
    ]
    if skip_if_webmap_exists and existing:
        status.update(state="SKIPPED", message="web map already exists", clone_id=existing[0].id)
        return {
            "suffix": suffix,
            "skipped": True,
            "webmap": existing[0],
            "status_file": str(status_path),
        }

    related_ids = extract_item_ids(source_webmap.get_data())
    fs_items = []
    for iid in sorted(related_ids):
        it = gis.content.get(iid)
        if it and it.type == "Feature Service":
            fs_items.append(it)
    if not fs_items:
        raise CloneValidationError("No Feature Service itemId found on source web map")
    if len(fs_items) != 1:
        raise CloneValidationError(
            f"Expected exactly 1 Feature Service on source web map, found {len(fs_items)}: "
            f"{[(it.title, it.id) for it in fs_items]}"
        )

    source_fs = fs_items[0]
    source_rows = layer_feature_counts(source_fs)
    try:
        source_summary = validate_feature_service_rows(
            f"source FS {source_fs.title!r}",
            source_rows,
            require_non_empty=True,
            min_layers=expected_fs_layers or 1,
        )
        if expected_fs_layers is not None and source_summary["total"] != expected_fs_layers:
            raise CloneValidationError(
                f"source FS has {source_summary['total']} layers, expected {expected_fs_layers}"
            )
    except CloneValidationError as exc:
        status.update(state="FAILED", error=str(exc), finished_at=_utc_now())
        raise

    if require_clean:
        incomplete = list_incomplete_fs_clones(gis, source_fs, folder=folder)
        if incomplete:
            details = [
                f"{it.title} ({it.id}) filled {summary['filled']}/{summary['total']}"
                for it, summary in incomplete
            ]
            msg = (
                "Incomplete/weird Feature Service clone(s) still present — "
                f"run cleanup first: {details}"
            )
            status.update(state="FAILED", error=msg, finished_at=_utc_now())
            raise CloneValidationError(msg)

    status.update(
        source_webmap_id=source_webmap.id,
        source_webmap_title=source_webmap.title,
        source_fs_ids=[it.id for it in fs_items],
        source_counts=source_summary,
    )

    cloned_fs = clone_feature_service_monitored(
        gis=gis,
        source_fs=source_fs,
        folder=folder,
        status=status,
        heartbeat_sec=heartbeat_sec,
        stall_timeout_sec=stall_timeout_sec,
    )
    rename_with_suffix(cloned_fs, suffix, source_title=source_fs.title)

    # Re-validate after rename
    assert_clone_matches_source(
        source_rows,
        layer_feature_counts(cloned_fs),
        label=f"renamed FS {cloned_fs.title!r}",
    )

    status.update(phase="clone_webmap", message="cloning web map with item_mapping")
    item_mapping = {source_fs.id: cloned_fs.id}

    webmap, all_items = clone_webmap_with_mapping(
        gis=gis,
        source_webmap=source_webmap,
        folder=folder,
        item_mapping=item_mapping,
        suffix=suffix,
    )
    if webmap.title != target_title:
        raise CloneValidationError(
            f"Unexpected web map title {webmap.title!r}, expected {target_title!r}"
        )

    try:
        report = assert_independent_webmap(source_webmap, webmap, cloned_fs)
    except CloneValidationError as exc:
        status.update(
            state="FAILED",
            error=str(exc),
            webmap_id=webmap.id,
            webmap_title=webmap.title,
            finished_at=_utc_now(),
        )
        raise

    status.update(
        state="SUCCESS",
        phase="done",
        webmap_id=webmap.id,
        webmap_title=webmap.title,
        verification=report,
        finished_at=_utc_now(),
    )
    return {
        "suffix": suffix,
        "skipped": False,
        "webmap": webmap,
        "feature_service": cloned_fs,
        "items": all_items,
        "report": report,
        "status_file": str(status_path),
        "folder": folder or "(root)",
    }
