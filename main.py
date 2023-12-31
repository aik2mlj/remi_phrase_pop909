from model import PopMusicTransformer
from datetime import datetime
import os
import argparse
import pickle
import utils
from finetune import load_split_file

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase", help="phrase configuration, eg. i4A8B8o4")
    parser.add_argument("-n", default=1, help="how many sample to generate")
    parser.add_argument("--only-melody", action="store_true")
    parser.add_argument("--prompt", help="the prompt midi path")
    parser.add_argument("--prompt-chord", help="the chord of prompt midi path")
    args = parser.parse_args()

    chkpt_name = 'REMI-PhBC-chord-melody' if args.only_melody else "REMI-PhBC-chord"
    phrase_configuration = utils.phrase_config_from_string(args.phrase)

    # declare model
    model = PopMusicTransformer(
        checkpoint=chkpt_name,
        is_training=False)
    
    if args.prompt is None:
        # generate from scratch
        for _ in range(int(args.n)):
            model.generate(
                phrase_configuration=phrase_configuration,
                temperature=1.2,
                topk=5,
                output_path=f"./result/gen({chkpt_name})-({args.phrase})_{datetime.now().strftime('%m-%d_%H%M%S')}.mid",
                prompt_paths=None)
    else:
        # generate continuation
        prompt_paths = {
            'midi_path': args.prompt,
            'melody_annotation_path': None,
            'chord_annotation_path': args.prompt_chord,
            'phrase_annotation_path': None,
        }
        prompt_id = args.prompt.split("/")[-1].split(".")[0].split("_")[-1]
        for _ in range(int(args.n)):
            model.generate(
                phrase_configuration=phrase_configuration,
                temperature=1.2,
                topk=5,
                output_path=f"./result/prompt_gen({chkpt_name})-({prompt_id})-({args.phrase})_{datetime.now().strftime('%m-%d_%H%M%S')}.mid",
                prompt_paths=prompt_paths)
    
    # close model
    model.close()

if __name__ == '__main__':
    main()
