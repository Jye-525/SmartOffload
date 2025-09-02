import math
from typing import Optional
from vllm.utils import cdiv
from vllm.v1.request import Request

class BaseReqsState:
    def __init__(self, reqs_state=None):
        if reqs_state is None:
            self.progress = {}
            self.finished_no = 0
            self.finished_mean = 0
            self.finished_M2 = 0
        else:
            self.progress = {req_id:reqs_state.progress[req_id] for (req_id,_) in reqs_state.running}
            # self.progress = reqs_state.progress.copy()
            self.finished_no = reqs_state.finished_no
            self.finished_mean = reqs_state.finished_mean
            self.finished_M2 = reqs_state.finished_M2

    def add_finished_req(self, num_tokens: int):
        self.finished_no += 1
        delta = num_tokens - self.finished_mean
        self.finished_mean += delta / self.finished_no
        delta2 = num_tokens - self.finished_mean
        self.finished_M2 += delta * delta2 # Welford's algorithm

    def update_req_progress(self, req_id: str, prompt_tokens: int, num_new_tokens: int, num_new_blocks: int):
        _, cmp_tokens, num_blocks = self.progress.get(req_id, (0, 0, 0))
        self.progress[req_id] = (prompt_tokens, cmp_tokens + num_new_tokens, num_blocks + num_new_blocks)

    # def ideal_new_tokens(self, req_id: str, prompt_tokens: int):
    #     _, cmp_tokens, _ = self.progress.get(req_id, (0, 0, 0))
    #     if cmp_tokens > prompt_tokens:
    #         return cmp_tokens - prompt_tokens
    #     else:
    #         return 1

    def estimate_output(self):
        if self.finished_no > 1:
            return math.ceil(math.fabs(self.finished_mean - math.sqrt(self.finished_M2 / (self.finished_no - 1))))
        return 1


class ReqsState(BaseReqsState):
    def __init__(self):
        self.running = []
        super().__init__()

    def add_running_req(self, req_id: str, prompt_tokens: int, is_renew: bool = False):
        self.running.append((req_id, prompt_tokens))
        # if not is_renew:
        #     ## init the progress for the new request
        #     self.progress[req_id] = (prompt_tokens, 0, 0)

class SimKVCache:
    def __init__(self, max_blocks: int, block_size: int, sim_cache=None):
        if sim_cache is None:
            self.max_blocks = max_blocks ### max available kv blocks
            self.allocated = 0
            self.block_size = block_size
            self.reqs_state = ReqsState()
        else:
            self.max_blocks = sim_cache.max_blocks
            self.allocated = sim_cache.allocated
            self.block_size = sim_cache.block_size
            self.reqs_state = BaseReqsState(sim_cache.reqs_state)

    def can_allocate(self, req_id: str, num_new_tokens: int):
        if req_id in self.reqs_state.progress:
            _, num_cur_tokens, num_cur_blocks = self.reqs_state.progress[req_id]
            new_blocks = cdiv(num_cur_tokens + num_new_tokens, self.block_size) - num_cur_blocks
        else:
            new_blocks = cdiv(num_new_tokens, self.block_size)
        
        # check if the new blocks can be allocated
        if self.allocated + new_blocks > self.max_blocks:
            return False
        return True
    
    def allocate(self, req_id: str, prompt_tokens: int, num_tokens: int, num_blocks: Optional[int] = None):
        if num_blocks is None:
            if req_id in self.reqs_state.progress:
                _, num_cur_tokens, num_cur_blocks = self.reqs_state.progress[req_id]
                num_blocks = cdiv(num_cur_tokens + num_tokens, self.block_size) - num_cur_blocks
            else:
                num_blocks = cdiv(num_tokens, self.block_size)
        self.allocated += num_blocks
        self.reqs_state.update_req_progress(req_id, prompt_tokens, num_tokens, num_blocks)

    def deallocate(self, req_id: str):
        assert req_id in self.reqs_state.progress, "Request not found in progress"
        self.allocated -= cdiv(self.reqs_state.progress[req_id][1], self.block_size)
        self.reqs_state.progress.pop(req_id)

    def available_tokens(self):
        return (self.max_blocks - self.allocated) * self.block_size

    def renew_running_reqs(self, reqs: list[Request]):
        self.reqs_state.running = []
        for req in reqs:
            self.reqs_state.add_running_req(req.request_id, req.num_prompt_tokens, is_renew=True)

    def add_running_reqs(self, reqs: list[Request]): 
        for req in reqs:
            self.reqs_state.add_running_req(req.request_id, req.num_prompt_tokens)

    def add_running_req(self, req_id: str, num_prompt_tokens: int):
        ## Note for preempted requests, num_prompt_tokens is the max of num_prompt_tokens and num_recomputed_tokens
        ### for new request, num_prompt_tokens is the num_prompt_tokens
        self.reqs_state.add_running_req(req_id, num_prompt_tokens)

    
    

    
        