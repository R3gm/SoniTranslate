import torch
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from vc_infer_pipeline import VC
import traceback, pdb
from lib.audio import load_audio
import numpy as np
import os
from fairseq import checkpoint_utils
import soundfile as sf
from gtts import gTTS
import edge_tts
import asyncio
import nest_asyncio

# model load
def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:  # change model or not
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ### if clean
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )



# inference
def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):  
    global tgt_sr, net_g, vc, hubert_model, version, cpt
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  # reemplace for 2
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=f0_file,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)



# hubert model
def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

# config cpu
def use_fp32_config():
    for config_file in [
        "32k.json",
        "40k.json",
        "48k.json",
        "48k_v2.json",
        "32k_v2.json",
    ]:
        with open(f"configs/{config_file}", "r") as f:
            strr = f.read().replace("true", "false")
        with open(f"configs/{config_file}", "w") as f:
            f.write(strr)

# config device and torch type
class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 2 # set cpu cores ####################
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16 series / 10 series graphics cards and P40 force single precision")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("Supported N-card not found, using MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, using CPU for inference")
            self.device = "cpu"
            self.is_half = False
            use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6GB VRAM configuration
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5GB VRAM configuration
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32




        print(self.device, self.is_half)

        return x_pad, x_query, x_center, x_max

# call inference
class ClassVoices:
    def __init__(self):
        self.file_index = "" # root

    def apply_conf(self, f0method,
                   model_voice_path00, transpose00, file_index2_00,
                   model_voice_path01, transpose01, file_index2_01,
                   model_voice_path02, transpose02, file_index2_02,
                   model_voice_path03, transpose03, file_index2_03,
                   model_voice_path04, transpose04, file_index2_04,
                   model_voice_path05, transpose05, file_index2_05,
                   model_voice_path99, transpose99, file_index2_99):

        #self.filename = filename
        self.f0method = f0method # pm
        
        self.model_voice_path00 = model_voice_path00
        self.transpose00 = transpose00
        self.file_index200 = file_index2_00

        self.model_voice_path01 = model_voice_path01
        self.transpose01 = transpose01
        self.file_index201 = file_index2_01

        self.model_voice_path02 = model_voice_path02
        self.transpose02 = transpose02
        self.file_index202 = file_index2_02

        self.model_voice_path03 = model_voice_path03
        self.transpose03 = transpose03
        self.file_index203 = file_index2_03

        self.model_voice_path04 = model_voice_path04
        self.transpose04 = transpose04
        self.file_index204 = file_index2_04

        self.model_voice_path05 = model_voice_path05
        self.transpose05 = transpose05
        self.file_index205 = file_index2_05

        self.model_voice_path99 = model_voice_path99
        self.transpose99 = transpose99
        self.file_index299 = file_index2_99
        return "CONFIGURATION APPLIED"

    def custom_voice(self,
        _values, # filter indices
        audio_files, # all audio files
        model_voice_path='',
        transpose=0,
        f0method='pm',
        file_index='',
        file_index2='',
        ):

        #hubert_model = None

        get_vc(
            sid=model_voice_path,  # model path
            to_return_protect0=0.33,
            to_return_protect1=0.33
        )

        for _value_item in _values:
            filename = "audio2/"+audio_files[_value_item] if _value_item != "test" else audio_files[0]
            #filename = "audio2/"+audio_files[_value_item]
            try:
                print(audio_files[_value_item], model_voice_path)
            except:
                pass

            info_, (sample_, audio_output_) = vc_single(
                sid=0,
                input_audio_path=filename, #f"audio2/{filename}",
                f0_up_key=transpose, # transpose for m to f and reverse 0 12
                f0_file=None,
                f0_method= f0method,
                file_index= file_index, # dir pwd?
                file_index2= file_index2,
                # file_big_npy1,
                index_rate= float(0.66),
                filter_radius= int(3),
                resample_sr= int(0),
                rms_mix_rate= float(0.25),
                protect= float(0.33),
            )

            sf.write(
                file= filename, #f"audio2/{filename}",
                samplerate=sample_,
                data=audio_output_
            )

        # detele the model

    def make_test(self, 
        tts_text, 
        tts_voice, 
        model_path,
        index_path,
        transpose,
        f0_method,
        ):
        os.system("rm -rf test")
        filename = "test/test.wav"

        if "SET_LIMIT" == os.getenv("DEMO"):
          if len(tts_text) > 60:
            tts_text = tts_text[:60]
            print("DEMO; limit to 60 characters")

        language = tts_voice[:2]
        try:
          os.system("mkdir test")
          #nest_asyncio.apply() # gradio;not
          asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
        except:
          try:
              tts = gTTS(tts_text, lang=language)
              tts.save(filename)
              tts.save
              print(f'No audio was received. Please change the tts voice for {tts_voice}. USING gTTS.')
          except:
            tts = gTTS('a', lang=language)
            tts.save(filename)
            print('Error: Audio will be replaced.')

        os.system("cp test/test.wav test/real_test.wav")

        self([],[]) # start modules

        self.custom_voice(
            ["test"], # filter indices
            ["test/test.wav"], # all audio files
            model_voice_path=model_path,
            transpose=transpose,
            f0method=f0_method,
            file_index='',
            file_index2=index_path,
        )
        return "test/test.wav", "test/real_test.wav"

    def __call__(self, speakers_list, audio_files):

        speakers_indices = {}

        for index, speak_ in enumerate(speakers_list):
            if speak_ in speakers_indices:
                speakers_indices[speak_].append(index)
            else:
                speakers_indices[speak_] = [index]

        
        # find models and index
        global weight_root, index_root, config, hubert_model
        weight_root = "weights"
        names = []
        for name in os.listdir(weight_root):
            if name.endswith(".pth"):
                names.append(name)

        index_root = "logs"
        index_paths = []
        for name in os.listdir(index_root):
            if name.endswith(".index"):
                index_paths.append(name)

        print(names, index_paths)
        # config machine
        hubert_model = None
        config = Config('cuda:0', is_half=True) # config = Config('cpu', is_half=False) # cpu

        # filter by speaker
        for _speak, _values in speakers_indices.items():
            #print(_speak, _values)
            #for _value_item in _values:
            #  self.filename = "audio2/"+audio_files[_value_item]
            ###print(audio_files[_value_item])

            #vc(_speak, _values, audio_files)

            if _speak == "SPEAKER_00":
              self.custom_voice(
                    _values, # filteredd
                    audio_files,
                    model_voice_path=self.model_voice_path00,
                    file_index2=self.file_index200,
                    transpose=self.transpose00,
                    f0method=self.f0method,
                    file_index=self.file_index,
                    )
            elif _speak == "SPEAKER_01":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path01,
                    file_index2=self.file_index201,
                    transpose=self.transpose01,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_02":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path02,
                    file_index2=self.file_index202,
                    transpose=self.transpose02,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_03":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path03,
                    file_index2=self.file_index203,
                    transpose=self.transpose03,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_04":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path04,
                    file_index2=self.file_index204,
                    transpose=self.transpose04,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_05":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path05,
                    file_index2=self.file_index205,
                    transpose=self.transpose05,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_99":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path99,
                    file_index2=self.file_index299,
                    transpose=self.transpose99,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            else:
                pass
