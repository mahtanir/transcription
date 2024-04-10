
import ctcdecode
import torch

class char_map_str:
    def __init__(self) -> None:
        self.char_map_str =  """
 ' 0
 <SPACE> 1
 a 2
 b 3
 c 4
 d 5
 e 6
 f 7
 g 8
 h 9
 i 10
 j 11
 k 12
 l 13
 m 14
 n 15
 o 16
 p 17
 q 18
 r 19
 s 20
 t 21
 u 22
 v 23
 w 24
 x 25
 y 26
 z 27
 """

labels = char_map_str().char_map_str
class CTCBeamDecoder: #use instead of greedy?

    def __init__(self, beam_size=100, blank_id=28, kenlm_path=None):
        print("loading beam search with lm...")
        
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels, alpha=0.522729216841, beta=0.96506699808,
            beam_width=beam_size, blank_id=28,
            model_path=kenlm_path) #change model path
        print("finished loading beam search")

    def __call__(self, output):
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])