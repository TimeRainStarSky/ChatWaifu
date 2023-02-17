import time
starttime = time.time()

import sys
out_path = sys.argv[1]
id = int(sys.argv[2])
text = ' '.join(sys.argv[3:])

if '--escape' in sys.argv:
  escape = True
else:
  escape = False

if id < 4:
  model = "CN/model.pth"
  config = "CN/config.json"
  text = "[ZH]" + text + "[ZH]"
elif id < 11:
  id = id - 4
  model = "JP/H_excluded.pth"
  config = "JP/config.json"
else:
  id = id - 11
  model = "JP/365_epochs.pth"
  config = "JP/config.json"

print('发言人：', id, '\n发言内容：', text, '\n正在加载模型：', model, sep='')

from scipy.io.wavfile import write
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import re
from torch import no_grad, LongTensor

hps_ms = utils.get_hparams_from_file(config)
n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

net_g_ms = SynthesizerTrn(
  n_symbols,
  hps_ms.data.filter_length // 2 + 1,
  hps_ms.train.segment_size // hps_ms.data.hop_length,
  n_speakers=n_speakers,
  emotion_embedding=emotion_embedding,
  **hps_ms.model)
_ = net_g_ms.eval()
utils.load_checkpoint(model, net_g_ms)

def get_label_value(text, label, default, warning_name='value'):
  value = re.search(rf'\[{label}=(.+?)\]', text)
  if value:
    try:
      text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
      value = float(value.group(1))
    except:
      print(f'Invalid {warning_name}!')
      sys.exit(1)
  else:
    value = default
  return value, text

def get_label(text, label):
  if f'[{label}]' in text:
    return True, text.replace(f'[{label}]', '')
  else:
    return False, text

if n_symbols != 0:
  if not emotion_embedding:
    if(1 == 1):
      choice = 't'
      if choice == 't':
        length_scale, text = get_label_value(
          text, 'LENGTH', 1, 'length scale')
        noise_scale, text = get_label_value(
          text, 'NOISE', 0.667, 'noise scale')
        noise_scale_w, text = get_label_value(
          text, 'NOISEW', 0.8, 'deviation of noise')
        cleaned, text = get_label(text, 'CLEANED')

        if cleaned:
          text_norm = text_to_sequence(text, hps_ms.symbols, [])
        else:
          text_norm = text_to_sequence(text, hps_ms.symbols, hps_ms.data.text_cleaners)
        if hps_ms.data.add_blank:
          text_norm = commons.intersperse(text_norm, 0)
        stn_tst = LongTensor(text_norm)

        with no_grad():
          x_tst = stn_tst.unsqueeze(0)
          x_tst_lengths = LongTensor([stn_tst.size(0)])
          sid = LongTensor([id])
          audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                       noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
      print('正在输出到文件：', out_path, sep='')
      write(out_path, hps_ms.data.sampling_rate, audio)

print('生成用时：', time.time() - starttime, '秒', sep='')