from datetime import datetime
import json
import math
import os
import sys

import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

from macaw.utils import SLOT_FROM_LC, GENERATOR_OPTIONS_DEFAULT, decompose_slots, get_raw_response, \
    load_model, make_input_from_example, run_model, run_model_with_outputs

MODEL_NAME_OR_PATH = "allenai/macaw-large"
LOG_FILE = "macaw_demo.log"  # Where to save a log
REST_API_PORT = None # Which port to use for API, set to None for no API

class Macaw(object):
	def __init__(self):
        self.model_dict = load_model(MODEL_NAME_OR_PATH, CUDA_DEVICES)
        MODEL, TOKENIZER, CUDA_DEVICE = self.model_dict['model'], self.model_dict['tokenizer'], self.model_dict['cuda_device']
        self.model = MODEL 
        self.tokenizer = TOKENIZER
        self.cuda_device = CUDA_DEVICE 

    def forward(self, in_string, out_strings):
    	in_string, out_strings = self.preprocess_in_out(in_string, out_strings)
    	return run_model_with_outputs(self.model, self.tokenizer, self.cuda_device, input_string, output_strings)

    def preprocess_in_out(self, in_string, out_strings):
    	return in_string, out_strings