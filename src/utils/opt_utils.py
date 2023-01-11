import os
import gc
import sys
import torch
import logging
from time import time
from uuid import uuid4
from itertools import zip_longest
from typing import List, Tuple, Dict, Union, NamedTuple, Optional, Iterable, cast
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

DEFAULT_STREAM_BATCH_SIZE = 0
DEFAULT_MAX_TOKENS = 36
DEFAULT_TEMPERATURE = 1.
DEFAULT_TOP_P = 1.
DEFAULT_NUM_RETURN_SEQUENCES = 1
EOS = '</s>'
DEFAULT_PROMPT = EOS
END_OF_STREAM = '[DONE]'
FINISH_REASON_EOS = 'stop'
FINISH_REASON_LENGTH = 'length'

OFFLOAD_DIR = os.path.join(os.environ.get('TORCH_HOME', '.cache'),  "opt_cache")
FIRST_GPU_MEMORY = "12GB"
SUCCESSIVE_GPU_MEMORY = "21GB"

MaxMemoryDict = Dict[Union[int, str], Union[int, str]]


class Completion(NamedTuple):
    text: str
    finish_reason: Optional[str]
    idx: int


class RawCompletion(NamedTuple):
    text: str
    pretruncation_num_new_tokens: int
    new_text: str
    truncated: bool


def clean_output_text(text: str) -> str:
    dirty_prefix = '</s>'
    return text[len(dirty_prefix):] if text.startswith(dirty_prefix) else text


def generate_response_id() -> str:
    return str(uuid4())


def get_timestamp() -> int:
    return int(time())


def truncate_at_stops(text: str, stop_strings: List[str]) -> Tuple[str, bool]:
    truncated = False
    for s in stop_strings:
        index = text.find(s)
        if index >= 0:
            text = text[:index]
            truncated = True
    return (text, truncated)


class OPTModel:
    def __init__(self, model_name='facebook/opt-66b', offload_dir=OFFLOAD_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
        max_memory = f'{free_in_GB - 2}GB'

        n_gpus = torch.cuda.device_count()
        self.max_memory = {i: max_memory for i in range(n_gpus)}
        self.max_memory = dict(
            (i, FIRST_GPU_MEMORY if i == 0 else SUCCESSIVE_GPU_MEMORY)
            for i in range(n_gpus)
        )
        print(f"MAX MEMORY: {self.max_memory}")


        self.main_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.offload_dir = offload_dir
        logging.info(f"OFFLOAD DIR: {OFFLOAD_DIR}")

        if self.offload_dir is not None:
            offload_state_dict = True
            if not os.path.isdir(self.offload_dir):
                logging.info(f'offload dir {self.offload_dir} does not exist; creating')
                try:
                    os.makedirs(self.offload_dir)
                except Exception as ex:
                    logging.warning(f'Could not create offload dir {self.offload_dir}', exc_info=ex)
        else:
            offload_state_dict = False

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            # load_in_8bit=True,
            max_memory=self.max_memory,
            torch_dtype=torch.float16,
            offload_folder=self.offload_dir,
            offload_state_dict=offload_state_dict,
        )

    def get_tokenizer_and_model(self):
        return self.tokenizer, self.model

    def complete(self, text: str,
                 # model_id: str,
                 stop_strings: List[str],
                 max_new_tokens: int = DEFAULT_MAX_TOKENS,
                 do_sample: bool = True,
                 top_p: float = DEFAULT_TOP_P,
                 temperature: float = DEFAULT_TEMPERATURE,
                 num_return_sequences: int = DEFAULT_NUM_RETURN_SEQUENCES,
                 **kwargs) -> List[Completion]:
        (tokenizer, model) = self.get_tokenizer_and_model()

        return [
            Completion(
                text=raw_completion.new_text,
                finish_reason=FINISH_REASON_EOS if raw_completion.truncated else FINISH_REASON_LENGTH,
                idx=i,
            )
            for (i, raw_completion) in enumerate(self._complete(
                text, tokenizer, model, stop_strings=stop_strings, max_new_tokens=max_new_tokens,
                do_sample=do_sample, top_p=top_p, temperature=temperature,
                num_return_sequences=num_return_sequences, **kwargs
            ))
        ]


    def stream_complete(
            self, text: str,
            # model_id: str,
            stop_strings: List[str],
            max_new_tokens: int = DEFAULT_MAX_TOKENS,
            do_sample: bool = True,
            top_p: float = DEFAULT_TOP_P,
            temperature: float = DEFAULT_TEMPERATURE,
            num_return_sequences: int = DEFAULT_NUM_RETURN_SEQUENCES,
            stream_batch_size: int = DEFAULT_STREAM_BATCH_SIZE) -> Iterable[Completion]:
        streams = [
            self._stream_complete_single(
                text,
                # model_id,
                stop_strings,
                max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p, temperature=temperature,
                index=i, stream_batch_size=stream_batch_size,
            )
            for i in range(num_return_sequences)
        ]
        return (
            c
            for c in (
            completion
            for completions in zip_longest(*streams, fillvalue=None)
            for completion in completions
        )
            if c is not None
        )

    def _stream_complete_single(
            self, text: str,
            # model_id: str,
            stop_strings: List[str],
            max_new_tokens: int = DEFAULT_MAX_TOKENS,
            do_sample: bool = True,
            top_p: float = DEFAULT_TOP_P,
            temperature: float = DEFAULT_TEMPERATURE,
            index: int = 0,
            stream_batch_size: int = DEFAULT_STREAM_BATCH_SIZE) -> Iterable[Completion]:
        (tokenizer, model) = self.get_tokenizer_and_model()

        prompt = text
        num_new_tokens = 0
        finish_reason = None
        while finish_reason is None:
            [raw_completion] = self._complete(
                text, tokenizer, model, stop_strings=stop_strings,
                max_new_tokens=min(
                    stream_batch_size if stream_batch_size > 0 else max_new_tokens,
                    max_new_tokens - num_new_tokens
                ),
                do_sample=do_sample, top_p=top_p, temperature=temperature,
                num_return_sequences=1,
            )

            if raw_completion.truncated:
                finish_reason = FINISH_REASON_EOS
            else:
                num_new_tokens += raw_completion.pretruncation_num_new_tokens
                if num_new_tokens >= max_new_tokens:
                    if num_new_tokens > max_new_tokens:
                        logging.warning('Generated more tokens than the max number specified')
                    finish_reason = FINISH_REASON_LENGTH

            # Check if a stop sequence spans the previous completion chunk and this one
            (truncated_text_after_prompt, truncated) = truncate_at_stops(
                raw_completion.text[len(prompt):],
                stop_strings)
            if truncated:
                truncation_index = len(prompt) + len(truncated_text_after_prompt)
                yield Completion(
                    text=raw_completion.text[-len(raw_completion.new_text):truncation_index],
                    finish_reason=FINISH_REASON_EOS,
                    idx=index,
                )
            else:
                yield Completion(text=raw_completion.new_text, finish_reason=finish_reason, idx=index)

            text = raw_completion.text

    def _complete(self, text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                  stop_strings: List[str], max_new_tokens: int,
                  do_sample: bool, top_p: float, temperature: float,
                  num_return_sequences: int, **kwargs) -> List[RawCompletion]:
        input_token_ids = tokenizer(text, return_tensors='pt')['input_ids']
        max_context = model.config.n_positions - max_new_tokens
        if hasattr(model.config, "n_positions") and input_token_ids.shape[1] > max_context:
            input_token_ids = input_token_ids[:, -max_context:]
            text = tokenizer.batch_decode(input_token_ids)[0]
            input_token_ids = tokenizer(text, return_tensors='pt')['input_ids']

        gen_out =model.generate(
            input_token_ids.to(self.main_device), max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p,
            temperature=temperature, num_return_sequences=num_return_sequences,
            # output_scores=True, return_dict_in_generate=True,
            **kwargs
        )

        output_token_ids = cast(torch.Tensor, gen_out)

        completions = []
        for completion_num in range(output_token_ids.shape[0]):
            output_text = clean_output_text(tokenizer.decode(output_token_ids[completion_num].tolist(),
                                                             clean_up_tokenization_spaces=False))
            if output_text.startswith(text):
                new_text = output_text[len(text):]
                (new_text, truncated) = truncate_at_stops(new_text, stop_strings)
                output_text = text + new_text
                completions.append(RawCompletion(
                    text=output_text,
                    pretruncation_num_new_tokens=output_token_ids.size(dim=1) - input_token_ids.size(dim=1),
                    new_text=new_text,
                    truncated=truncated,
                ))
            else:
                breakpoint()
                raise Exception(f'Generated text "{output_text}" does not begin with input text "{text}"')

        return completions


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    text = """
QUESTION:
Rocks are classified as igneous, metamorphic, or sedimentary according to
ANSWER:
how they formed
HYPOTHESIS:
rocks are classified as igneous, sedimentary, or metamorphic based on how they are formed

QUESTION:
Which two body systems are directly involved in movement?
ANSWER:
muscular and skeletal
HYPOTHESIS:
the muscular system and the skeletal system are involved in helping an animal move

QUESTION:
Which change in the state of water particles causes the particles to become arranged in a fixed position?
ANSWER:
freezing
HYPOTHESIS:
freezing can change the particles in water to an orderly fixed position

QUESTION:
Which of the following was probably most important in the formation of dark, fertile soil that is good for farming?
ANSWER:
plant decomposition
HYPOTHESIS:
plant decomposition is important for the formation of fertile soil

QUESTION:
When igneous rock is changed into metamorphic rock, which form of energy is this process?
ANSWER:
heat
HYPOTHESIS:
heat energy is in the process of changing igneous rock to metamorphic rock

QUESTION:
Which of these items contains only a solution?
ANSWER:
a bottle of juice
HYPOTHESIS:
the bottle contains a solution

QUESTION:
Many natural rock formations change color over time. In Utah, for example, iron oxidized and formed red, orange, and yellow rock. Which of the following is the cause of this change?
ANSWER:
chemical weathering
HYPOTHESIS:
chemical weathering can cause iron in rock to oxidize and change the color of the rock to yellow, red, or orange

QUESTION:
Automobile engines built today are designed to be gas efficient. Gas-efficient engines most likely affect a city by reducing
ANSWER:
air pollution.
HYPOTHESIS:
using a gas-efficient engine in automobiles will cause less air pollution

QUESTION:
A student standing near a campfire feels warmer as the fire grows. Which process most likely transfers heat from the campfire to the student?
ANSWER:
radiation
HYPOTHESIS:
the campfire transfers heat through waves in the process of radiation

QUESTION:
In snapdragon plants, red flowers are dominant to white flowers. When red-flowered plants are crossed with white-flowered plants, pink flowers are produced. If a researcher wants to produce plants with only white flowers, what color should the parent plants be?
ANSWER:
both white
HYPOTHESIS:
crossing two snapdragon plants with white flower will cause the offspring to have white flowers

QUESTION:
Soil erosion can be best prevented by
ANSWER:
building terraces into the sides of a slope.
HYPOTHESIS:
building terraces on a slope prevents soil erosion on that slope

QUESTION:
Which of the following properties provides the BEST way to identify a mineral?
ANSWER:
hardness
HYPOTHESIS:
hardness can be used to identify minerals

QUESTION:
When ice cream is left out of a freezer, the ice cream changes from a ___.
ANSWER:
solid to a liquid
HYPOTHESIS:
the ice cream will melt and change from a solid to a liquid

QUESTION:
Which equipment will best separate a mixture of iron filings and black pepper?
ANSWER:
magnet
HYPOTHESIS:
magnet can be used to separate a mixture of iron fillings and black pepper

QUESTION:
When a transverse wave passes from right to left through a medium, what happens to the particles of the medium?
ANSWER:
Particles move back and forth perpendicular to the wave.
HYPOTHESIS:
transverse waves cause particles to move perpendicular to the direction of the wave

QUESTION:
Decomposers increase the fertility of the soil and prevent dead organisms from building up in the environment. In which way do decomposers make the soil more fertile?
ANSWER:
by adding nitrogen
HYPOTHESIS:
decomposers make fertile soil by adding nitrogen to soil

QUESTION:
When oxygen combines with hydrogen, which substance is formed?
ANSWER:
water
HYPOTHESIS:
oxygen combining with hydrogen will form water

QUESTION:
Which body system is most responsible for the removal of waste?
ANSWER:
excretory system
HYPOTHESIS:
the excretory system removes waste from the body

QUESTION:
In which environment is white fur color an advantage for survival?
ANSWER:
arctic tundra
HYPOTHESIS:
an organism having white fur has a positive impact on that organism's survival in arctic environments

QUESTION:
Which of these will best separate iron filings from sand?
ANSWER:
a magnet
HYPOTHESIS:
a magnet can be used to separate iron fillings from sand

QUESTION:
Which best describes transportation technology?
ANSWER:
a system that is used to move people and products
HYPOTHESIS:
transportation technology is a system that moves people and products

QUESTION:
Drew was measuring the growth of a vine that can grow almost 31 cm a day. Which would be the best way to record his data of the growth over a period of a day?
ANSWER:
a line graph
HYPOTHESIS:"""


    text_10 = """QUESTION:
Rocks are classified as igneous, metamorphic, or sedimentary according to
ANSWER:
how they formed
HYPOTHESIS:
rocks are classified as igneous, sedimentary, or metamorphic based on how they are formed

QUESTION:
Which two body systems are directly involved in movement?
ANSWER:
muscular and skeletal
HYPOTHESIS:
the muscular system and the skeletal system are involved in helping an animal move

QUESTION:
Which change in the state of water particles causes the particles to become arranged in a fixed position?
ANSWER:
freezing
HYPOTHESIS:
freezing can change the particles in water to an orderly fixed position

QUESTION:
Which of the following was probably most important in the formation of dark, fertile soil that is good for farming?
ANSWER:
plant decomposition
HYPOTHESIS:
plant decomposition is important for the formation of fertile soil

QUESTION:
When igneous rock is changed into metamorphic rock, which form of energy is this process?
ANSWER:
heat
HYPOTHESIS:
heat energy is in the process of changing igneous rock to metamorphic rock

QUESTION:
Which of these items contains only a solution?
ANSWER:
a bottle of juice
HYPOTHESIS:
the bottle contains a solution

QUESTION:
Many natural rock formations change color over time. In Utah, for example, iron oxidized and formed red, orange, and yellow rock. Which of the following is the cause of this change?
ANSWER:
chemical weathering
HYPOTHESIS:
chemical weathering can cause iron in rock to oxidize and change the color of the rock to yellow, red, or orange

QUESTION:
Automobile engines built today are designed to be gas efficient. Gas-efficient engines most likely affect a city by reducing
ANSWER:
air pollution.
HYPOTHESIS:
using a gas-efficient engine in automobiles will cause less air pollution

QUESTION:
A student standing near a campfire feels warmer as the fire grows. Which process most likely transfers heat from the campfire to the student?
ANSWER:
radiation
HYPOTHESIS:
the campfire transfers heat through waves in the process of radiation

QUESTION:
Soil erosion can be best prevented by
ANSWER:
building terraces into the sides of a slope.
HYPOTHESIS:
building terraces on a slope prevents soil erosion on that slope

QUESTION:
Drew was measuring the growth of a vine that can grow almost 31 cm a day. Which would be the best way to record his data of the growth over a period of a day?
ANSWER:
a line graph
HYPOTHESIS:"""

    model = OPTModel()
    sttime = time()
    print("generating....")
    print(model.complete("this is a test. ", stop_strings=['QUESTION'], top_p=0.8, num_return_sequences=1))
    print(model.complete(text, stop_strings=['QUESTION'], top_p=0.8, num_return_sequences=1))
    print(f"generation took {time() - sttime} seconds")
    print("generating 2...")
    print(model.complete(text_10, stop_strings=['QUESTION'], top_p=0.8, num_return_sequences=1))
    print(f"generation took {time() - sttime} seconds")