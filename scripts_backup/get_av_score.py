import os
from argparse import ArgumentParser
import json

def parser_args():
    parser = ArgumentParser()
    parser.add_argument('--input')
    return parser.parse_args()


def get_av_score(file):
    with open(file, 'r') as f:
        data = json.load(f)
    # calculate average score
    results = data['results']
    total = 0
    for dataset in results.keys():
        total += results[dataset]['acc']
    av_score = total / len(results)
    print(f"average score: {av_score}")
    # save score
    filename = os.path.join(os.path.dirname(args.input), 'av_score.txt')
    with open(filename, 'w') as f:
        f.write(str(av_score))
    print(f"saved in {filename}")
    return av_score

def main(args):
    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.endswith('.json'):
                get_av_score(os.path.join(args.input, file))
    else:
        get_av_score(args.input)


if __name__ == '__main__':
    args = parser_args()
    main(args)
