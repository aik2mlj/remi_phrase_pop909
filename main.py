from model import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-PhBC-chord',
        is_training=False)
    
    # generate from scratch
    phrase_configuration = [('i', 4), ('A', 8), ('B', 8), ('o', 4)]  # your decision
    model.generate(
        phrase_configuration=phrase_configuration,
        temperature=1.2,
        topk=5,
        output_path='./result/from_scratch.midi',
        prompt=None)

    # close model
    model.close()

if __name__ == '__main__':
    main()
