from model import PopMusicTransformer
from datetime import datetime
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-melody", action="store_true")
    args = parser.parse_args()

    chkpt_name = 'REMI-PhBC-chord-melody' if args.only_melody else "REMI-PhBC-chord"

    # declare model
    model = PopMusicTransformer(
        checkpoint=chkpt_name,
        is_training=False)
    
    # generate from scratch
    phrase_configuration = [('i', 4), ('A', 8), ('B', 8), ('o', 4)]  # your decision

    model.generate(
        phrase_configuration=phrase_configuration,
        temperature=1.2,
        topk=5,
        output_path=f"./result/gen({chkpt_name})_{datetime.now().strftime('%m-%d_%H%M')}.midi",
        prompt=None)

    # close model
    model.close()

if __name__ == '__main__':
    main()
