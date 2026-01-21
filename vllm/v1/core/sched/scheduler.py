# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Optional, Union
import copy

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.utils import cdiv, sha256
from vllm.v1.stats_utils.sched_stats import SchedStatsCollector
from vllm.v1.core.sched.simulate_kv_cache import RequestsOracle, SimKVCache, ReqsState

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.sched_stats = SchedStatsCollector.get()
        print(f"Scheduler get sched_stats: {self.sched_stats}")

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.include_finished_set = include_finished_set

        self.schedule_method = envs.VLLM_V1_SCHEDULE_METHOD
        self.dy_change = envs.VLLM_V1_ADAPTIVE_MAX_NUM_REQS # for test only
        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        # if self.schedule_method == "default":
        #    self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        # else: 
        #     ### cost-aware-resume: Decide if schedule the resume request based on KV cache usage and resume window
        #     ### evict-optimal: Decide if schedule the resume request based on the number of scheduled running request compared with last step.
        #     ## To support dynamic chaning of max_num_running_reqs in a iteration
        #     if self.dy_change:
        #         self.max_num_running_reqs_lower_bound = min(256, self.scheduler_config.max_num_seqs)  
        #         self.max_num_running_reqs_upper_bound = max(self.max_num_running_reqs_lower_bound, self.scheduler_config.max_num_seqs)
        #         self.max_num_running_reqs = self.max_num_running_reqs_lower_bound
        #     else:
        #         self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        #         self.max_num_running_reqs_lower_bound = self.max_num_running_reqs
        #         self.max_num_running_reqs_upper_bound = self.max_num_running_reqs 
        
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            self.connector = KVConnectorFactory.create_connector_v1(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> deque of CachedRequestData
        self._cached_reqs_data: dict[
            str, deque[CachedRequestData]] = defaultdict(deque)

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats)
        
        self.schedule_step_count = 0
        logger.info(f"Scheduler initialized with method: {self.schedule_method}, max_kv_cache_blocks={kv_cache_config.num_blocks}, max_free_kv_blocks={self.kv_cache_manager.block_pool.get_num_free_blocks()}")
        # Bogdan's approach to decide if resume the request or not
        self.num_last_scheduled_running_reqs = 0
        
        # record average-output len (for all the requests system-wide)
        self.finished_reqs_count = 0
        self.total_output_len = 0
        self.avg_output_len = 1
        
        # Used to count the importance of each skip in opt-5
        self.opt5_skip_count_1 = 0
        self.opt5_skip_count_2 = 0
        self.opt5_skip_count_3 = 0
        
        # track the est_total_blocks for each request (used in opt-6-4)
        self.req_to_est_total_blocks: dict[str, int] = {}
        
        #######
        # # Enhanced adaptive threshold management
        self.min_kv_threshold = 0.85  # Conservative lower bound
        self.max_kv_threshold = 1  # Aggressive upper bound
        self.base_kv_threshold = self.max_kv_threshold  # Starting point
        self.adaptive_kv_threshold = self.base_kv_threshold
        
        # Performance tracking for adaptation
        self.preemption_kv_usage_history = deque(maxlen=50)  # Track recent KV usage
        self.threshold_update_frequency = 10  # How often to adjust the threshold
        
        self.estimate_output_len: dict[str, int] = {}  # request_id -> estimated output length
        # self.reqs_state = ReqsState() # used by opt-7 to track all requests
        if self.schedule_method == "evict-optimal-7" and envs.VLLM_V1_OUTPUT_LENGTH_PREDICTOR=="ideal":
             assert envs.VLLM_V1_DATASETS_ORACLE_FILE is not None, "Please provide VLLM_V1_DATASETS_ORACLE_FILE for ideal output length predictor"
             requests_oracle = RequestsOracle()
             requests_oracle.initialize(envs.VLLM_V1_DATASETS_ORACLE_FILE)

        self.sim_kv_cache = SimKVCache(max_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(), block_size=self.block_size)

    def schedule(self) -> SchedulerOutput:
        if self.schedule_method == "default":
            return self._schedule_default()
        elif self.schedule_method == "evict-optimal-1":
            # schedule the resumed/new resquest based on if there are enough kv cache 
            # for prefill or recomputed tokens of the resumed/new request 
            return self._schedule_evict_optimal_1()
        elif self.schedule_method == "kv-usage-aware":
            ## fixed version of kv-usage-aware when scheduling preempted requests
            return self._schedule_evict_optimal_5()
        elif self.schedule_method == "kv-usage-aware-1":
            ## use the freeness feature to schedule both preempted and new requests
            return self._schedule_evict_optimal_5_1()
        elif self.schedule_method == "evict-optimal-7":
            return self._schedule_evict_optimal_7()

    def _schedule_default(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        total_preempted_tokens = 0
        total_resumed_tokens = 0

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        self.schedule_step_count += 1

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when PP>1 and
                #    we have already scheduled all prompt tokens but they are
                #    not finished yet.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # print(f"Step {self.schedule_step_count} schedule running request {request.request_id} pt: {request.num_prompt_tokens}, progress: {request.num_computed_tokens}, to schedule: {num_new_tokens}, KV available: {self.kv_cache_manager.block_pool.get_num_free_blocks()}, token_budget: {token_budget}")

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request. (remove the most recently last added request)
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    total_preempted_tokens += preempted_req.num_computed_tokens
                    preempted_req.num_recomputed_tokens = preempted_req.num_computed_tokens
                    preempted_req.last_preempted_step_id = self.schedule_step_count
                    preempted_req.repeat_preempt_count += 1
                    preempted_req.preempt_step_ids.append(self.schedule_step_count)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0

                    # update scheduled_stats to record the preempted request
                    self.sched_stats.record_preempted(self.schedule_step_count, [
                        {
                            "request_id": preempted_req.request_id,
                            "preempted_tokens": preempted_req.num_recomputed_tokens,
                        }
                    ])
                    # print(f"Step {self.schedule_step_count} preempted request {preempted_req.request_id} pt: {preempted_req.num_prompt_tokens}, to preempt: {preempted_req.num_recomputed_tokens}, KV available: {self.kv_cache_manager.block_pool.get_num_free_blocks()}")

                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # print(f"======Step {self.schedule_step_count} schedule running request {request.request_id} pt: {request.num_prompt_tokens}, progress: {request.num_computed_tokens}, to schedule: {num_new_tokens}, KV available: {self.kv_cache_manager.block_pool.get_num_free_blocks()}, token_budget: {token_budget}")


            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting[0]

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if self.lora_config and request.lora_request and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    continue

                # Get already-cached tokens.
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(
                        request)

                # Get externally-cached tokens if using a KVConnector.
                num_external_tokens = (
                    0 if self.connector is None else
                    self.connector.get_num_new_matched_tokens(
                        request, num_computed_tokens))

                # Total computed tokens (local + external).
                num_computed_tokens += num_external_tokens

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_budget) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_budget)
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break
                else:
                    encoder_inputs_to_schedule = None
                    new_encoder_budget = encoder_budget

                # print(f"Step {self.schedule_step_count} schedule waiting request {request.request_id}, status: {request.status}, pt: {request.num_prompt_tokens}, progress: {request.num_computed_tokens}, to schedule: {num_new_tokens}, KV available: {self.kv_cache_manager.block_pool.get_num_free_blocks()}, token_budget: {token_budget}, max_blocks: {self.kv_cache_manager.num_gpu_blocks} max_blocks_1: {self.kv_cache_manager.block_pool.num_gpu_blocks}")

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_tokens,
                    computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break
                
                # print(f"======Step {self.schedule_step_count} schedule waiting request {request.request_id}, status: {request.status}, pt: {request.num_prompt_tokens}, progress: {request.num_computed_tokens}, to schedule: {num_new_tokens}, blocks: {len(new_blocks)}, token_budget: {token_budget}")
                # KVConnector: update internal state after allocation.
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        num_external_tokens,
                    )

                self.waiting.popleft()
                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                    total_resumed_tokens += request.num_recomputed_tokens
                    request.resume_step_ids.append(self.schedule_step_count)
                    ## update scheduled_stats to record the resumed request
                    self.sched_stats.record_resumed(self.schedule_step_count, [
                        {
                            "request_id": request.request_id,
                            "resumed_tokens": request.num_recomputed_tokens,
                        }
                    ])
                    ### reset the preempt status
                    request.last_preempted_step_id = -1
                    request.last_try_resume_step_id = -1
                    request.num_recomputed_tokens = 0
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            len(self.running),
        )
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        ### Update scheduled_stats to record the new scheduled reqs, running reqs, and resumed reqs
        self.sched_stats.record_newly_scheduled(self.schedule_step_count, [
            req.request_id for req in scheduled_new_reqs
        ])

        self.sched_stats.record_running(self.schedule_step_count, [
            req.request_id for req in scheduled_running_reqs
        ])

        self.sched_stats.add_metrics(self.schedule_step_count, {
            "total_preempted_tokens": total_preempted_tokens,
            "total_resumed_tokens": total_resumed_tokens,
        })

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        # if total_preempted_tokens > 0:
        #     logger.info(f"Scheduler preempted {len(preempted_reqs)} requests with {total_preempted_tokens} tokens on step {self.schedule_step_count}")
        # if total_resumed_tokens > 0:
        #     logger.info(f"Scheduler scheduled {len(resumed_reqs_data)} resumed requests with {total_resumed_tokens} tokens on step {self.schedule_step_count}")
        
        return scheduler_output
     
    def _schedule_evict_optimal_1(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        total_preempted_tokens = 0
        total_resumed_tokens = 0

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        self.schedule_step_count += 1

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            if len(scheduled_running_reqs) == self.max_num_running_reqs:
                break
            
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when PP>1 and
                #    we have already scheduled all prompt tokens but they are
                #    not finished yet.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request. (remove the most recently last added request)
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    total_preempted_tokens += preempted_req.num_computed_tokens
                    preempted_req.num_recomputed_tokens = preempted_req.num_computed_tokens
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.last_preempted_step_id = self.schedule_step_count
                    preempted_req.repeat_preempt_count += 1
                    preempted_req.preempt_step_ids.append(self.schedule_step_count)
                    preempted_req.num_computed_tokens = 0 

                    # update scheduled_stats to record the preempted request
                    self.sched_stats.record_preempted(self.schedule_step_count, [
                        {
                            "request_id": preempted_req.request_id,
                            "preempted_tokens": preempted_req.num_recomputed_tokens,
                        }
                    ])

                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        cur_scheduled_running_reqs = len(scheduled_running_reqs)
        
        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()
        
        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) >= self.max_num_running_reqs:
                    break

                request = self.waiting[0]

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if self.lora_config and request.lora_request and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    continue

                # Get already-cached tokens.
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(
                        request)

                # Get externally-cached tokens if using a KVConnector.
                num_external_tokens = (
                    0 if self.connector is None else
                    self.connector.get_num_new_matched_tokens(
                        request, num_computed_tokens))

                # Total computed tokens (local + external).
                num_computed_tokens += num_external_tokens

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_budget) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_budget)
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break
                else:
                    encoder_inputs_to_schedule = None
                    new_encoder_budget = encoder_budget

                # check if the resumed request can be scheduled
                # Method 4: based on if the available KV cache have enough room for the prefill/recomputed tokens of current request
                if request.status == RequestStatus.PREEMPTED:
                    tmp_req_tokens = max(request.num_prompt_tokens, request.num_recomputed_tokens)
                else:
                    tmp_req_tokens = request.num_prompt_tokens
                # Check If there are not request exit compared with last iteration, the request will be died again if resumed 
                num_required_blocks = cdiv(tmp_req_tokens, self.kv_cache_manager.block_size)
                if num_required_blocks > self.kv_cache_manager.block_pool.get_num_free_blocks():
                    break
                
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_tokens,
                    computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVConnector: update internal state after allocation.
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        num_external_tokens,
                    )

                self.waiting.popleft()
                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                    total_resumed_tokens += request.num_recomputed_tokens
                    request.resume_step_ids.append(self.schedule_step_count)

                    # update scheduled_stats to record the resumed request
                    self.sched_stats.record_resumed(self.schedule_step_count, [
                        {
                            "request_id": request.request_id,
                            "resumed_tokens": request.num_recomputed_tokens,
                        }
                    ])

                    # resumed successfully, reset the preempt/resume step_id
                    request.last_preempted_step_id = -1
                    request.last_try_resume_step_id = -1
                    request.num_recomputed_tokens = 0
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # assert len(self.running) <= self.max_num_running_reqs_upper_bound
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        total_scheduled_reqs = (len(scheduled_new_reqs) +
                                len(scheduled_resumed_reqs) +
                                len(scheduled_running_reqs))
        assert total_scheduled_reqs <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            len(self.running),
        )
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        ### Update scheduled_stats to record the new scheduled reqs, running reqs, and resumed reqs
        self.sched_stats.record_newly_scheduled(self.schedule_step_count, [
            req.request_id for req in scheduled_new_reqs
        ])

        self.sched_stats.record_running(self.schedule_step_count, [
            req.request_id for req in scheduled_running_reqs
        ])

        self.sched_stats.add_metrics(self.schedule_step_count, {
            "total_preempted_tokens": total_preempted_tokens,
            "total_resumed_tokens": total_resumed_tokens,
        })

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        
        # check if need to change the self.max_num_running_reqs for the next step
        # Current idea, change it based on the current KV cache memory utilization.
        assert self.dy_change == False, "Dynamic change of max_num_running_reqs is not supported in evict optimal scheduler"
            
        ### update the scheduled_running_reqs
        self.num_last_scheduled_running_reqs = cur_scheduled_running_reqs
        
        # if total_preempted_tokens > 0:
        #     logger.info(f"Scheduler preempted {len(preempted_reqs)} requests with {total_preempted_tokens} tokens on step {self.schedule_step_count}")
        # if total_resumed_tokens > 0:
        #     logger.info(f"Scheduler scheduled {len(resumed_reqs_data)} resumed requests with {total_resumed_tokens} tokens on step {self.schedule_step_count}")
        
        return scheduler_output
    
    def _schedule_evict_optimal_5(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        total_preempted_tokens = 0
        total_resumed_tokens = 0

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        self.schedule_step_count += 1

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            if len(scheduled_running_reqs) == self.max_num_running_reqs:
                break
            
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when PP>1 and
                #    we have already scheduled all prompt tokens but they are
                #    not finished yet.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request. (remove the most recently last added request)
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    total_preempted_tokens += preempted_req.num_computed_tokens
                    preempted_req.num_recomputed_tokens = preempted_req.num_computed_tokens
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.last_preempted_step_id = self.schedule_step_count
                    preempted_req.repeat_preempt_count += 1
                    preempted_req.preempt_step_ids.append(self.schedule_step_count)
                    preempted_req.num_computed_tokens = 0

                    # update scheduled_stats to record the preempted request
                    self.sched_stats.record_preempted(self.schedule_step_count, [
                        {
                            "request_id": preempted_req.request_id,
                            "preempted_tokens": preempted_req.num_recomputed_tokens,
                        }
                    ])

                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        cur_scheduled_running_reqs = len(scheduled_running_reqs)
        
        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()
        
        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) >= self.max_num_running_reqs:
                    break

                request = self.waiting[0]

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if self.lora_config and request.lora_request and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    continue

                # Get already-cached tokens.
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(
                        request)

                # Get externally-cached tokens if using a KVConnector.
                num_external_tokens = (
                    0 if self.connector is None else
                    self.connector.get_num_new_matched_tokens(
                        request, num_computed_tokens))

                # Total computed tokens (local + external).
                num_computed_tokens += num_external_tokens

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_budget) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_budget)
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break
                else:
                    encoder_inputs_to_schedule = None
                    new_encoder_budget = encoder_budget

                ################### Only for resumed requests ########################
                ## method 1 only for resumed requests 
                if request.status == RequestStatus.PREEMPTED:
                    if self.num_last_scheduled_running_reqs <= cur_scheduled_running_reqs:
                        self.opt5_skip_count_1 += 1
                        # no request is exit
                        break
                    # check the free memory can support the estimated output tokens
                    est_tokens = max(request.num_prompt_tokens, request.num_recomputed_tokens)
                    num_required_blocks = cdiv(est_tokens, self.kv_cache_manager.block_size)
                    if num_required_blocks > self.kv_cache_manager.block_pool.get_num_free_blocks():
                        self.opt5_skip_count_2 += 1
                        break
                    else:
                        est_kv_usage = 1.0 - ((self.kv_cache_manager.block_pool.get_num_free_blocks() - num_required_blocks) / self.kv_cache_manager.num_gpu_blocks)
                        if est_kv_usage > 0.90:
                            self.opt5_skip_count_3 += 1
                            break
                
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_tokens,
                    computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVConnector: update internal state after allocation.
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        num_external_tokens,
                    )

                self.waiting.popleft()
                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                    total_resumed_tokens += request.num_recomputed_tokens
                    request.resume_step_ids.append(self.schedule_step_count)

                    # update scheduled_stats to record the resumed request
                    self.sched_stats.record_resumed(self.schedule_step_count, [
                        {
                            "request_id": request.request_id,
                            "resumed_tokens": request.num_recomputed_tokens,
                        }
                    ])

                    # resumed successfully, reset the preempt/resume step_id
                    request.last_preempted_step_id = -1
                    request.last_try_resume_step_id = -1
                    request.num_recomputed_tokens = 0
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # assert len(self.running) <= self.max_num_running_reqs_upper_bound
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        total_scheduled_reqs = (len(scheduled_new_reqs) +
                                len(scheduled_resumed_reqs) +
                                len(scheduled_running_reqs))
        assert total_scheduled_reqs <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            len(self.running),
        )
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        ### Update scheduled_stats to record the new scheduled reqs, running reqs, and resumed reqs
        self.sched_stats.record_newly_scheduled(self.schedule_step_count, [
            req.request_id for req in scheduled_new_reqs
        ])

        self.sched_stats.record_running(self.schedule_step_count, [
            req.request_id for req in scheduled_running_reqs
        ])
        
        self.sched_stats.add_metrics(self.schedule_step_count, {
            "total_preempted_tokens": total_preempted_tokens,
            "total_resumed_tokens": total_resumed_tokens,
        })

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        
        # check if need to change the self.max_num_running_reqs for the next step
        # Current idea, change it based on the current KV cache memory utilization.
        assert self.dy_change == False, "Dynamic change of max_num_running_reqs is not supported in evict optimal scheduler"
            
        ### update the scheduled_running_reqs
        self.num_last_scheduled_running_reqs = cur_scheduled_running_reqs
        
        # if total_preempted_tokens > 0:
        #     logger.info(f"Scheduler preempted {len(preempted_reqs)} requests with {total_preempted_tokens} tokens on step {self.schedule_step_count}")
        # if total_resumed_tokens > 0:
        #     logger.info(f"Scheduler scheduled {len(resumed_reqs_data)} resumed requests with {total_resumed_tokens} tokens on step {self.schedule_step_count}  ; skip_count_1 = {self.opt5_skip_count_1} skip_count_2 = {self.opt5_skip_count_2} skip_count_3 = {self.opt5_skip_count_3}")
        
        return scheduler_output
    
    def _schedule_evict_optimal_5_1(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        total_preempted_tokens = 0
        total_resumed_tokens = 0

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        self.schedule_step_count += 1

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            if len(scheduled_running_reqs) == self.max_num_running_reqs:
                break
            
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when PP>1 and
                #    we have already scheduled all prompt tokens but they are
                #    not finished yet.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request. (remove the most recently last added request)
                    cur_kv_usage = self._get_current_kv_usage()
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    total_preempted_tokens += preempted_req.num_computed_tokens
                    preempted_req.num_recomputed_tokens = preempted_req.num_computed_tokens
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.last_preempted_step_id = self.schedule_step_count
                    preempted_req.repeat_preempt_count += 1
                    preempted_req.preempt_step_ids.append(self.schedule_step_count)
                    preempted_req.num_computed_tokens = 0 

                    # update scheduled_stats to record the preempted request
                    self.sched_stats.record_preempted(self.schedule_step_count, [
                        {
                            "request_id": preempted_req.request_id,
                            "preempted_tokens": preempted_req.num_recomputed_tokens,
                        }
                    ])

                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    self._update_adaptive_kv_threshold(
                        preemption_occurred=True,
                        current_kv_usage=cur_kv_usage,
                    )
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        cur_scheduled_running_reqs = len(scheduled_running_reqs)
        
        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()
        
        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            #### check if the freeness is < 1, if so, skip scheduling from waiting queue
            skip_schedule = False
            freeness = self._get_freeness(cur_scheduled_running_reqs)
            if freeness < 1:
                skip_schedule = True
            
            while self.waiting and token_budget > 0 and not skip_schedule:
                if len(self.running) >= self.max_num_running_reqs:
                    break

                request = self.waiting[0]

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if self.lora_config and request.lora_request and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    continue

                # Get already-cached tokens.
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(
                        request)

                # Get externally-cached tokens if using a KVConnector.
                num_external_tokens = (
                    0 if self.connector is None else
                    self.connector.get_num_new_matched_tokens(
                        request, num_computed_tokens))

                # Total computed tokens (local + external).
                num_computed_tokens += num_external_tokens

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_budget) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_budget)
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break
                else:
                    encoder_inputs_to_schedule = None
                    new_encoder_budget = encoder_budget

                ################### Only for resumed requests ########################
                ## method 1 only for resumed requests 
                if request.status == RequestStatus.PREEMPTED:
                    if self.num_last_scheduled_running_reqs <= cur_scheduled_running_reqs:
                        break
                    # check the free memory can support the estimated output tokens
                    new_prefill_tokens = max(request.num_prompt_tokens, request.num_recomputed_tokens)
                    num_required_blocks = cdiv(new_prefill_tokens, self.kv_cache_manager.block_size)
                    if num_required_blocks > self.kv_cache_manager.block_pool.get_num_free_blocks():
                        break
                    else:
                        ## check if the freeness is < 1 once this request is scheduled
                        freeness = self._get_freeness(cur_scheduled_running_reqs + len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + 1, num_required_blocks)
                        if freeness < 1:
                            break
                else:
                    #### for new requests, check if the freeness is < 1 once this request is scheduled
                    full_pt_required_blocks = cdiv(num_new_tokens, self.kv_cache_manager.block_size)
                    if full_pt_required_blocks > self.kv_cache_manager.block_pool.get_num_free_blocks():
                        break
                    freeness = self._get_freeness(cur_scheduled_running_reqs + len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + 1, full_pt_required_blocks)
                    if freeness < 1:
                        break
                
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_tokens,
                    computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVConnector: update internal state after allocation.
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        num_external_tokens,
                    )

                self.waiting.popleft()
                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                    total_resumed_tokens += request.num_recomputed_tokens
                    request.resume_step_ids.append(self.schedule_step_count)

                    # update scheduled_stats to record the resumed request
                    self.sched_stats.record_resumed(self.schedule_step_count, [
                        {
                            "request_id": request.request_id,
                            "resumed_tokens": request.num_recomputed_tokens,
                        }
                    ])

                    # resumed successfully, reset the preempt/resume step_id
                    request.last_preempted_step_id = -1
                    request.last_try_resume_step_id = -1
                    request.num_recomputed_tokens = 0
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # assert len(self.running) <= self.max_num_running_reqs_upper_bound
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        total_scheduled_reqs = (len(scheduled_new_reqs) +
                                len(scheduled_resumed_reqs) +
                                len(scheduled_running_reqs))
        assert total_scheduled_reqs <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            len(self.running),
        )
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        ### Update scheduled_stats to record the new scheduled reqs, running reqs, and resumed reqs
        self.sched_stats.record_newly_scheduled(self.schedule_step_count, [
            req.request_id for req in scheduled_new_reqs
        ])

        self.sched_stats.record_running(self.schedule_step_count, [
            req.request_id for req in scheduled_running_reqs
        ])
        
        self.sched_stats.add_metrics(self.schedule_step_count, {
            "total_preempted_tokens": total_preempted_tokens,
            "total_resumed_tokens": total_resumed_tokens,
        })

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        
        # check if need to change the self.max_num_running_reqs for the next step
        # Current idea, change it based on the current KV cache memory utilization.
        assert self.dy_change == False, "Dynamic change of max_num_running_reqs is not supported in evict optimal scheduler"
            
        ### update the scheduled_running_reqs
        self.num_last_scheduled_running_reqs = cur_scheduled_running_reqs
        if len(scheduled_new_reqs) > 0:
            ## update the kv threshold when there is no preempted request to resume
            cur_kv_usage = self._get_current_kv_usage()
            self._update_adaptive_kv_threshold(
                preemption_occurred=False,
                current_kv_usage=cur_kv_usage,
            )
        
        # if total_preempted_tokens > 0:
        #     logger.info(f"Scheduler preempted {len(preempted_reqs)} requests with {total_preempted_tokens} tokens on step {self.schedule_step_count}")
        # if total_resumed_tokens > 0:
        #     logger.info(f"Scheduler scheduled {len(resumed_reqs_data)} resumed requests with {total_resumed_tokens} tokens on step {self.schedule_step_count}  ; skip_count_1 = {self.opt5_skip_count_1} skip_count_2 = {self.opt5_skip_count_2} skip_count_3 = {self.opt5_skip_count_3}")
        
        return scheduler_output
     
    def _schedule_evict_optimal_7(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        total_preempted_tokens = 0
        total_resumed_tokens = 0

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        self.schedule_step_count += 1

        # print(f"Before schedule requests at step {self.schedule_step_count}, free_blocks: {self.kv_cache_manager.block_pool.get_num_free_blocks()}")
        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            if len(scheduled_running_reqs) == self.max_num_running_reqs:
                break
            
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when PP>1 and
                #    we have already scheduled all prompt tokens but they are
                #    not finished yet.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue
            
            available_tokens = self.sim_kv_cache.available_tokens()
            if available_tokens > 0:
                num_new_tokens = min(num_new_tokens, available_tokens)

            # logger.info(f"Scheduler-Real: req {request.request_id} progress at step {self.schedule_step_count}: prmpt={request.num_prompt_tokens}, comp={request.num_computed_tokens}, to_schedule={num_new_tokens}, free_kv_blocks={self.kv_cache_manager.block_pool.get_num_free_blocks()}")
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request. (remove the most recently last added request)
                    preempted_req = self.running.pop()
                    ## update the simulate kv cache to deallocate blocks for the preempted request
                    # self.sim_kv_cache.deallocate(preempted_req.request_id, len(self.kv_cache_manager.req_to_blocks[preempted_req.request_id]))
                    self.sim_kv_cache.deallocate(preempted_req.request_id)
                    ## free real allocated blocks for the preempted request
                    self.kv_cache_manager.free(preempted_req)
                    total_preempted_tokens += preempted_req.num_computed_tokens
                    preempted_req.num_recomputed_tokens = preempted_req.num_computed_tokens
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.last_preempted_step_id = self.schedule_step_count
                    preempted_req.repeat_preempt_count += 1
                    preempted_req.preempt_step_ids.append(self.schedule_step_count)
                    preempted_req.num_computed_tokens = 0
                    
                    #### update scheduled_stats to record the preempted request
                    self.sched_stats.record_preempted(self.schedule_step_count, [
                        {
                            "request_id": preempted_req.request_id,
                            "preempted_tokens": preempted_req.num_recomputed_tokens,
                        }
                    ])

                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            # update the request progress in reqs_state for opt-7
            self.sim_kv_cache.allocate(request.request_id, request.num_prompt_tokens, num_new_tokens, len(new_blocks))
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        cur_scheduled_running_reqs = len(scheduled_running_reqs)
        # (TODO):update reqs_state's running requests list for opt-7
        self.sim_kv_cache.renew_running_reqs(scheduled_running_reqs)

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()
        
        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) >= self.max_num_running_reqs:
                    break

                request = self.waiting[0]

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if self.lora_config and request.lora_request and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    continue

                # Get already-cached tokens.
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(
                        request)

                # Get externally-cached tokens if using a KVConnector.
                num_external_tokens = (
                    0 if self.connector is None else
                    self.connector.get_num_new_matched_tokens(
                        request, num_computed_tokens))

                # Total computed tokens (local + external).
                num_computed_tokens += num_external_tokens

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_budget) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_budget)
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break
                else:
                    encoder_inputs_to_schedule = None
                    new_encoder_budget = encoder_budget

                ################### For both preempted and new requests, preempted requests are handled first ########################
                ## Step 1: check if the free kv cache blocks are enough for num_new_tokens tokens (assert num_external_tokens == 0)
                ##(TODO): handle the case when num_external_tokens > 0 in the future
                if not self._simulate_running_queue(request, num_new_tokens):
                    break
                
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_tokens,
                    computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVConnector: update internal state after allocation.
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        num_external_tokens,
                    )

                self.waiting.popleft()
                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                self.running.append(request)
                
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                    ##### update the simulate kv cache to add the request, and allocate blocks for the request
                    self.sim_kv_cache.add_running_req(request.request_id, request.num_prompt_tokens)
                    self.sim_kv_cache.allocate(request.request_id, request.num_prompt_tokens, num_new_tokens, len(new_blocks))
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                    ##### update the simulate kv cache to add the request, and allocate blocks for the request
                    self.sim_kv_cache.add_running_req(request.request_id, max(request.num_prompt_tokens, request.num_recomputed_tokens))
                    # logger.info(f"+++++++++++Scheduler-Opt7: Resuming preempted req {request.request_id} at step {self.schedule_step_count}: prmpt={request.num_prompt_tokens}, recomputed={request.num_recomputed_tokens}, to_schedule={num_new_tokens}")
                    self.sim_kv_cache.allocate(request.request_id, request.num_prompt_tokens, num_new_tokens, len(new_blocks))
                    ### update statistics info
                    total_resumed_tokens += request.num_recomputed_tokens
                    request.resume_step_ids.append(self.schedule_step_count)
                    self.sched_stats.record_resumed(self.schedule_step_count, [
                        {
                            "request_id": request.request_id,
                            "resumed_tokens": request.num_recomputed_tokens,
                        }
                    ])
                    # resumed successfully, reset the preempt/resume step_id
                    request.last_preempted_step_id = -1
                    request.last_try_resume_step_id = -1
                    request.num_recomputed_tokens = 0
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]

                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # assert len(self.running) <= self.max_num_running_reqs_upper_bound
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        total_scheduled_reqs = (len(scheduled_new_reqs) +
                                len(scheduled_resumed_reqs) +
                                len(scheduled_running_reqs))
        assert total_scheduled_reqs <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            len(self.running),
        )
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        ### Update scheduled_stats to record the new scheduled reqs, running reqs, and resumed reqs
        self.sched_stats.record_newly_scheduled(self.schedule_step_count, [
            req.request_id for req in scheduled_new_reqs
        ])
        self.sched_stats.record_running(self.schedule_step_count, [
            req.request_id for req in scheduled_running_reqs
        ])

        self.sched_stats.add_metrics(self.schedule_step_count, {
            "total_preempted_tokens": total_preempted_tokens,
            "total_resumed_tokens": total_resumed_tokens,
        })

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        
        # check if need to change the self.max_num_running_reqs for the next step
        # Current idea, change it based on the current KV cache memory utilization.
        assert self.dy_change == False, "Dynamic change of max_num_running_reqs is not supported in evict optimal scheduler"
            
        ### update the scheduled_running_reqs
        self.num_last_scheduled_running_reqs = cur_scheduled_running_reqs
        
        # if total_preempted_tokens > 0:
        #     logger.info(f"Scheduler preempted {len(preempted_reqs)} requests with {total_preempted_tokens} tokens on step {self.schedule_step_count}")
        # if total_resumed_tokens > 0:
        #     logger.info(f"Scheduler scheduled {len(resumed_reqs_data)} resumed requests with {total_resumed_tokens} tokens on step {self.schedule_step_count} ; skip_count_1 = {self.opt5_skip_count_1}")
        
        return scheduler_output
 
    def _make_cached_request_data(
        self,
        request: Request,
        num_scheduled_tokens: int,
        num_scheduled_spec_tokens: int,
        new_block_ids: list[int],
        resumed_from_preemption: bool,
    ) -> CachedRequestData:
        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        num_computed_tokens = request.num_computed_tokens
        num_regular_tokens = num_scheduled_tokens - num_scheduled_spec_tokens
        new_token_ids = request.all_token_ids[
            num_computed_tokens:num_computed_tokens + num_regular_tokens]

        req_data_queue = self._cached_reqs_data.get(request.request_id)
        if req_data_queue:
            req_data = req_data_queue.popleft()
            req_data.resumed_from_preemption = resumed_from_preemption
            req_data.new_token_ids = new_token_ids
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
        else:
            # No cached request data, or all cached request data has been
            # used by the scheduled requests.
            req_data = CachedRequestData.from_request(request,
                                                      resumed_from_preemption,
                                                      new_token_ids,
                                                      new_block_ids)
        return req_data

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        new_running: list[Request] = []
        outputs: list[EngineCoreOutput] = []
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens, where is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # OPTIMIZATION: Avoid list(set) if the set is empty.
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    mm_positions = request.mm_positions[input_id]
                    start_pos = mm_positions.offset
                    num_tokens = mm_positions.length
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                request.spec_token_ids = spec_token_ids[req_index]

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids

            # Append generated tokens and check for stop. Note that if
            # a request is still being prefilled, we expect the model runner
            # to return empty token ids for the request.
            for num_new, output_token_id in enumerate(new_token_ids, 1):
                request.append_output_token_ids(output_token_id)

                # Check for stop and update request state.
                # This must be called before we make the EngineCoreOutput.
                stopped = check_stop(request, self.max_model_len)
                if stopped:
                    self._free_request(request)
                    del new_token_ids[num_new:]  # Trim new tokens if needed.
                    break

            # Extract sample logprobs if needed.
            if request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and request.use_structured_output:
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids:
                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        pred_out_length=request.pred_output_length,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        preempt_info=request.get_preempt_info()))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            if not stopped:
                new_running.append(request)

        # Return the cached request data to the queue so they can be reused.
        for req_data in scheduler_output.scheduled_cached_reqs:
            self._cached_reqs_data[req_data.req_id].append(req_data)

        self.running = new_running
        engine_core_outputs = EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(spec_decoding_stats),
        )
        if self.include_finished_set:
            #TODO currently sending duplicates here, improve this
            engine_core_outputs.finished_requests = (
                scheduler_output.finished_req_ids | self.finished_req_ids)

        return engine_core_outputs

    def add_request(self, request: Request) -> None:
        logger.debug(f"Scheduler add new request {request.request_id} to the waiting queue with prompt length {request.num_prompt_tokens}")
        self.waiting.append(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        ### update the simulate kv cache to deallocate blocks
        if self.schedule_method == "evict-optimal-7":
            if request.status == RequestStatus.FINISHED_STOPPED or request.status == RequestStatus.FINISHED_LENGTH_CAPPED:
                self.sim_kv_cache.reqs_state.add_finished_req(request.num_output_tokens)
            self.sim_kv_cache.deallocate(request.request_id)
        
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        self.encoder_cache_manager.free(request)
        self._cached_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats
    
    def _update_adaptive_kv_threshold(self, preemption_occurred: bool, current_kv_usage: float):
        """Dynamically adjust KV threshold based on system performance"""
        if preemption_occurred:
            # Too aggressive - increase threshold to be more conservative
            self.adaptive_kv_threshold = max(0.85, self.adaptive_kv_threshold - 0.02)
            self.preemption_kv_usage_history.append(current_kv_usage)
        else:
            # System stable - can be more aggressive
            self.adaptive_kv_threshold = min(1.0, self.adaptive_kv_threshold + 0.01)
        
        # Update every N steps based on recent preemption patterns
        if len(self.preemption_kv_usage_history) >= self.threshold_update_frequency:
            avg_preemption_usage = sum(self.preemption_kv_usage_history) / len(self.preemption_kv_usage_history)
            # Set threshold slightly below average preemption point
            self.adaptive_kv_threshold = max(0.85, avg_preemption_usage - 0.05)
            self.preemption_kv_usage_history.clear()

    def _get_current_kv_usage(self) -> float:
        """Calculate current KV cache usage as a percentage."""
        total_blocks = self.kv_cache_manager.num_gpu_blocks
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        return 1.0 - (free_blocks / total_blocks)

    def _get_freeness(self, num_running_reqs, new_allocated=0) -> float:
        ## get total blocks and free blocks
        total_blocks = self.kv_cache_manager.num_gpu_blocks
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if num_running_reqs > 0:
            return (free_blocks - new_allocated) / num_running_reqs
        else:
            return total_blocks

    # def _simulate_next_step(self, sim_kv_cache_copy: SimKVCache) -> bool:
    #     context = 0
    #     new_running = []
    #     for (req_id, pt) in sim_kv_cache_copy.reqs_state.running:
    #         progress = sim_kv_cache_copy.reqs_state.progress.get(req_id, 0)
    #         est_output = sim_kv_cache_copy.reqs_state.estimate_output(req_id)
    #         if progress == pt + est_output:
    #             sim_kv_cache_copy.reqs_state.add_finished(req_id, est_output)
    #             sim_kv_cache_copy.deallocate(req_id)
    #             continue
    #         new_tokens = min(self.max_num_scheduled_tokens - context, sim_kv_cache_copy.reqs_state.ideal_new_tokens(req_id, pt))
    #         if not sim_kv_cache_copy.can_allocate(req_id, new_tokens):
    #             return False
    #         context += new_tokens
    #         new_running.append((req_id, pt))
    #     sim_kv_cache_copy.reqs_state.running = new_running
    #     return True
    
    def _simulate_running_queue(self, request: Request, num_new_tokens: int) -> bool:
        # copy the simulate kv cache
        sim_kv_cache_copy = SimKVCache(self.sim_kv_cache.max_blocks, self.block_size, self.sim_kv_cache)
        # check if can allocate blocks for the given request
        if not sim_kv_cache_copy.can_allocate(request.request_id, num_new_tokens):
            return False
        
        ## allocate blocks for the given request
        sim_kv_cache_copy.allocate(request.request_id, request.num_prompt_tokens, num_new_tokens)

        if request.status == RequestStatus.PREEMPTED:
            new_prompt_tokens = max(request.num_prompt_tokens, request.num_recomputed_tokens)
        else:
            new_prompt_tokens = request.num_prompt_tokens

        # Build sim_running
        # sim_running = {req_id: (pt, max(1, pt + sim_kv_cache_copy.reqs_state.estimate_output() - sim_kv_cache_copy.reqs_state.progress[req_id][0])) for (req_id, pt) in self.sim_kv_cache.reqs_state.running}
        sim_running = {}
        for (req_id, pt) in self.sim_kv_cache.reqs_state.running:
            est_outlen = sim_kv_cache_copy.reqs_state.estimate_output(req_id)
            sim_running[req_id] = (pt, max(1, pt + est_outlen - sim_kv_cache_copy.reqs_state.progress[req_id][0]))
            if envs.VLLM_V1_OUTPUT_LENGTH_PREDICTOR=="ideal":
                self.requests[req_id].update_pred_output_length(est_outlen)  # Initial update
            else:    
                self.requests[req_id].update_pred_output_length(sim_kv_cache_copy.reqs_state.progress[req_id][1] - sim_kv_cache_copy.reqs_state.progress[req_id][0] + est_outlen)  # Initial update
                logger.info(f"++++++Scheduler-Opt7: Initial running req {req_id} update predicted output length. Progress: {sim_kv_cache_copy.reqs_state.progress[req_id][1] - sim_kv_cache_copy.reqs_state.progress[req_id][0]}. est_output={est_outlen} finished reqs so far: {sim_kv_cache_copy.reqs_state.finished_reqs}")
        
        cur_est_output = sim_kv_cache_copy.reqs_state.estimate_output(request.request_id)
        self.requests[request.request_id].update_pred_output_length(cur_est_output)
        cur_remaining_tokens = new_prompt_tokens + cur_est_output - num_new_tokens
        cur_remaining_pt = new_prompt_tokens - sim_kv_cache_copy.reqs_state.progress[request.request_id][1]
        ## sort the sim running requests by the remaining tokens
        sim_sorted = sorted(sim_running.items(), key=lambda x: x[1][1])

        remaining_prefill_iters = 0
        cmp_remaining_pt = 0
        iteration = 0
        idx = 0
        while idx < len(sim_sorted):
            # if cur_remaining_tokens - iteration == 0:
            (req_id, (prompt_tokens, remaining_tokens)) = sim_sorted[idx]
            if cur_remaining_tokens - cmp_remaining_pt - (iteration - remaining_prefill_iters) == 0:
                return True
            # (req_id, (prompt_tokens, remaining_tokens)) = sim_sorted[idx]
            # update predicted output length for the request
            if envs.VLLM_V1_OUTPUT_LENGTH_PREDICTOR !="ideal": 
                self.requests[req_id].update_pred_output_length(sim_kv_cache_copy.reqs_state.progress[req_id][1] - sim_kv_cache_copy.reqs_state.progress[req_id][0])
            
            fast_forward = min(remaining_tokens, cur_remaining_tokens) - iteration
            if fast_forward == 0:
                # (TODO): should we add this request to the finished requests?
                output_tokens = sim_kv_cache_copy.reqs_state.progress[req_id][1] - sim_kv_cache_copy.reqs_state.progress[req_id][0]
                sim_kv_cache_copy.reqs_state.add_finished_req(output_tokens)
                sim_kv_cache_copy.deallocate(req_id)
                idx += 1
                continue
            
            for running_req_idx in range(idx, len(sim_sorted)):
                if not sim_kv_cache_copy.can_allocate(sim_sorted[running_req_idx][0], fast_forward):
                    return False
                sim_kv_cache_copy.allocate(sim_sorted[running_req_idx][0], sim_sorted[running_req_idx][1][0], fast_forward)
            
            ### check if fast_forward is enough to schedule all the remaining prefill tokens
            max_prefill_iters = cdiv(cur_remaining_pt, (self.max_num_scheduled_tokens - len(sim_sorted))) 
            try_alloc_tokens = 0
            if max_prefill_iters < fast_forward:
                try_alloc_tokens = cur_remaining_pt + (fast_forward - max_prefill_iters)
                cmp_remaining_pt = cur_remaining_pt
                remaining_prefill_iters = max_prefill_iters 
                cur_remaining_pt = 0
            else:
                try_alloc_tokens = (self.max_num_scheduled_tokens - len(sim_sorted)) * fast_forward
                cmp_remaining_pt += try_alloc_tokens
                remaining_prefill_iters += fast_forward 
                cur_remaining_pt -= try_alloc_tokens

            if not sim_kv_cache_copy.can_allocate(request.request_id, try_alloc_tokens):
                return False
            sim_kv_cache_copy.allocate(request.request_id, request.num_prompt_tokens, try_alloc_tokens)
            iteration += fast_forward
        return True
    