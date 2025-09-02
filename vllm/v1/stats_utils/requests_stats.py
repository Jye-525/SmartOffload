import csv
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import vllm.envs as envs


@dataclass
class RequestRecord:
    request_id: str
    arrival_time_s: float = 0.0 # wall-clock time
    num_prompt_tokens: int = 0
    num_gen_tokens: int = 0
    e2e_latency_ms: Optional[float] = None
    queued_time_ms: Optional[float] = None
    prefill_time_ms: Optional[float] = None
    decode_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None
    ttft_latency_ms: Optional[float] = None
    tpot_latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    preempt_count: int = 0
    preempt_step_ids: List[int] = field(default_factory=list)
    resume_step_ids: List[int] = field(default_factory=list)


class RequestStatsCollector:
    ## only scheduler call this class, it it thread-safe
    _instance: Optional["RequestStatsCollector"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self, enable: bool, csv_path: Optional[str] = None) -> None:
        self._enabled: bool = bool(enable)
        self._records: Dict[str, RequestRecord] = {}
        self._written_header: bool = False
        
        self._csv_path = csv_path
        # Ensure directory exists if enabled
        if self._enabled:
            assert self._csv_path is not None, "csv_path is required"
            print(f"SchedStatsCollector: enabled, csv_path: {self._csv_path}")
            Path(self._csv_path).parent.mkdir(parents=True, exist_ok=True)

    # ----- Singleton accessors -----
    @staticmethod
    def get() -> "RequestStatsCollector":
        inst = RequestStatsCollector._instance
        if inst is None:
            with RequestStatsCollector._instance_lock:
                inst = RequestStatsCollector._instance
                if inst is None:
                    if envs.VLLM_V1_REQ_STATS_ENABLED:
                        inst = RequestStatsCollector(
                            enable=True, csv_path=envs.VLLM_V1_REQ_STATS_CSV_PATH 
                        )
                    else:
                        inst = _NoOpRequestStatsCollector()
                    RequestStatsCollector._instance = inst
        return inst

    @staticmethod
    def get_instance() -> "RequestStatsCollector":
        # Back-compat alias for get()
        return RequestStatsCollector.get()

    def is_enabled(self) -> bool:
        return self._enabled

    def record_finished_request(self, request_id: str, metrics: Dict[str, Any]):
        if not self._enabled:
            return
        
        record = self._get_or_create(request_id)
        for key, value in metrics.items():
            self._apply_metric(record, key, value)

        # calculate ttft and tpot latency
        if record.ttft_latency_ms is None:
            record.ttft_latency_ms = record.queued_time_ms + record.prefill_time_ms
        if record.tpot_latency_ms is None:
            record.tpot_latency_ms = record.decode_time_ms / record.num_gen_tokens
            
    def finalize(self):
        if not self._enabled or not self._records:
            return
        
        # with self._lock:
        self._flush()

    ###### Internal interface ######
    @staticmethod
    def _header_fields() -> list[str]:
        ## return static csv header fields
        return [
            "request_id",
            "arrival_time_s",
            "num_prompt_tokens",
            "num_gen_tokens",
            "e2e_latency_ms",
            "queued_time_ms",
            "prefill_time_ms",
            "decode_time_ms",
            "inference_time_ms",
            "ttft_latency_ms",
            "tpot_latency_ms",
            "finish_reason",
            "preempt_count",
            "preempt_step_ids",
            "resume_step_ids",
        ]
    
    def _record_to_row(self, record: RequestRecord) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "request_id": record.request_id,
            "arrival_time_s": record.arrival_time_s,
            "num_prompt_tokens": record.num_prompt_tokens,
            "num_gen_tokens": record.num_gen_tokens,
            "e2e_latency_ms": record.e2e_latency_ms,
            "queued_time_ms": record.queued_time_ms,
            "prefill_time_ms": record.prefill_time_ms,
            "decode_time_ms": record.decode_time_ms,
            "inference_time_ms": record.inference_time_ms,
            "ttft_latency_ms": record.ttft_latency_ms,
            "tpot_latency_ms": record.tpot_latency_ms,
            "finish_reason": record.finish_reason,
            "preempt_count": record.preempt_count,
            "preempt_step_ids": record.preempt_step_ids,
            "resume_step_ids": record.resume_step_ids,
        }

        return row
    
    def _flush(self) -> None:
        if not self._enabled:
            return
        
        header = self._header_fields()
        path = Path(self._csv_path)

        write_header = not self._written_header or not path.exists()
        rows = [self._record_to_row(record) for key, record in self._records.items()]

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
                self._written_header = True
            for row in rows:
                writer.writerow(row)

        # Clear buffer after flushing
        self._records.clear()

    def _get_or_create(self, request_id: str) -> RequestRecord:
        if request_id not in self._records:
            self._records[request_id] = RequestRecord(request_id=request_id)
        return self._records[request_id]

    def _apply_metric(self, record: RequestRecord, key: str, value: Any) -> None:
        # Known direct attributes
        if hasattr(record, key):
            setattr(record, key, value)
        else:
            print(f"Unknown metric: {key}, value: {value}")


class _NoOpRequestStatsCollector(RequestStatsCollector):
    def __init__(self) -> None:
        super().__init__(enable=False)

    def is_enabled(self) -> bool:
        return False

    def record_finished_request(self, metrics: Dict[str, Any]) -> None:
        pass

    def update_preempt_info(self, request_id: str, preempt_count: int, preempt_step_ids: List[int], resume_step_ids: List[int]) -> None:
        pass

    def finalize(self) -> None:
        pass