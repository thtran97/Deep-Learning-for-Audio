import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import onnx

from src.model import Encoder
from src.dataset import BaseLoad
from src.utils import zcr_vad, get_timestamp
from src.cluster import OptimizedAgglomerativeClustering

# from openvino.inference_engine import IECore, IENetwork

class BasePredictor(BaseLoad):
    def __init__(self, config_path, max_frame, hop):
        config = torch.load(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(config.sr, config.n_mfcc)
        self.ndim = config.ndim
        self.max_frame = max_frame
        self.hop = hop
        
    @staticmethod
    def _plot_diarization(y, spans, speakers):
        c = y[0].cpu().numpy().copy()
        for (start, end), speaker in zip(spans, speakers):
            c[start:end] = speaker
 
        plt.figure(figsize=(15, 2))
        plt.plot(y[0], "k-")
        for idx, speaker in enumerate(set(speakers)):
            plt.fill_between(range(len(c)), -1, 1, where=(c==speaker), alpha=0.5, label=f"speaker_{speaker}")
        plt.legend(loc="upper center", ncol=idx+1, bbox_to_anchor=(0.5, -0.25))
        
        
class PyTorchPredictor(BasePredictor):
    def __init__(self, config_path, model_path, max_frame=45, hop=3):
        super().__init__(config_path, max_frame, hop)
        
        weight = torch.load(model_path, map_location="cpu")
        self.model = Encoder(self.ndim).to(self.device)
        self.model.load_state_dict(weight)
        self.model.eval()
    
    def predict(self, path, plot=False):        
        y = self._load(path, mfcc=False)
#         print("Length of y: ",y.shape)
        activity = zcr_vad(y)
        spans = get_timestamp(activity)
        
        embed = [self._encode_segment(y, span) for span in spans]
        embed = torch.cat(embed).cpu().numpy()
        print("Embed shape: ", embed.shape)
        speakers = OptimizedAgglomerativeClustering().fit_predict(embed)
        
        if plot:
            self._plot_diarization(y, spans, speakers)
            
        timestamp = np.array(spans) / self.sr
        return timestamp, speakers
    
    def _encode_segment(self, y, span):
        start, end = span
        mfcc = self._mfcc(y[:, start:end]).to(self.device)
#         print("Size of mfcc :",mfcc.size())
        mfcc = mfcc.unfold(2, self.max_frame, self.hop).permute(2, 0, 1, 3)
#         print("Size of unfold mfcc: ", mfcc.size())
        with torch.no_grad():
            embed = self.model(mfcc).mean(0, keepdims=True)
#         print("Size of embedded vector from mfcc: ", embed.size())
        return embed
        
    def to_onnx(self, outdir="model/openvino"):
        os.makedirs(outdir, exist_ok=True)
        mfcc = torch.rand(1, 1, self.n_mfcc, self.max_frame).to(self.device)
        onnx.export(self.model, mfcc, f"{outdir}/diarization.onnx", input_names=["input"], output_names=["output"])
        print(f"model is exported as {outdir}/diarization.onnx")     
        

class OpenVINOPredictor(BasePredictor):
    def __init__(self, model_xml, model_bin, config_path, max_frame=30, hop=3):
        super().__init__(config_path, max_frame, hop)
        net = IENetwork(model_xml, model_bin)

        plugin = IECore()
        self.exec_net = plugin.load_network(net, "CPU")

    def predict(self, path, plot=False):        
        y = self._load(path, mfcc=False)
        activity = zcr_vad(y)
        spans = get_timestamp(activity)
        
        embed = [self._encode_segment(y, span) for span in spans]
        embed = np.vstack(embed)
        speakers = OptimizedAgglomerativeClustering().fit_predict(embed)
        
        if plot:
            self._plot_diarization(y, spans, speakers)
            
        timestamp = np.array(spans) / self.sr
        return timestamp, speakers
    
    def _encode_segment(self, y, span):
        start, end = span
        mfcc = self._mfcc(y[:, start:end])
        mfcc = mfcc.unfold(2, self.max_frame, self.hop).permute(2, 0, 1, 3)
        mfcc = mfcc.cpu().numpy()
        embed = [self.exec_net.infer({"input": m}) for m in mfcc]
        embed = np.array([e["output"] for e in embed])
        embed = embed.mean(0)
        return embed        