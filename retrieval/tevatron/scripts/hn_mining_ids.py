import random
from abc import ABC, abstractmethod
from tqdm import tqdm
from argparse import ArgumentParser


class HardNegatives(ABC):
    def __init__(self, qrels_path, run_path, out_path):
        self.load_qrels(qrels_path)
        self.load_run(run_path)
        self.out_path = out_path

    @abstractmethod
    def choose_negatives(self, query: int, n: int) -> list[str]:
        # each subclass implements the way to sample the negatives for the query
        pass

    def load_qrels(self, path):
        self.qid2pos = {}
        with open(path, 'r') as h:
            for line in h:
                qid, q0, did, rel = line.strip().split("\t")
                if qid not in self.qid2pos:
                    self.qid2pos[qid] = []
                self.qid2pos[qid].append(did)

    def load_run(self, path):
        self.qid2negs = {}
        with open(path, 'r') as h:
            for line in h:
                qid, did, score = line.strip().split()
                if qid not in self.qid2negs:
                    self.qid2negs[qid] = []
                if did not in self.qid2pos[qid]:
                    self.qid2negs[qid].append(did) 
    
    def sample_hard_negatives(self, n):
        with open(self.out_path, 'w') as h:
            for query in tqdm(self.qid2negs):
                
                chosen_negatives = self.choose_negatives(query, n)
                
                line_to_write = f"{query}\t{','.join(chosen_negatives)}\n"
                h.write(line_to_write)

class RandomHardNegatives(HardNegatives):

    def set_seed(self, seed):
        random.seed(seed)

    def choose_negatives(self, query: int, n: int) -> list[str]:
        possible_negatives = self.qid2negs[query]
        return random.sample(possible_negatives, n)

class RandomHardNegativesWithMomentum(HardNegatives):

    def __init__(self, qrels_path, run_path, out_path, prev_run_path):
        super().__init__(qrels_path, run_path, out_path)
        self.load_prev_run(prev_run_path)

    def load_prev_run(self, path):
        self.qid2prev_negs = {}
        with open(path, 'r') as h:
            for line in h:
                qid, did, score = line.strip().split()
                if qid not in self.qid2prev_negs:
                    self.qid2prev_negs[qid] = []
                if did not in self.qid2pos[qid]:
                    self.qid2prev_negs[qid].append(did) 

    def set_seed(self, seed):
        random.seed(seed)

    def choose_negatives(self, query: int, n: int) -> list[str]:
        current_negative_pool = self.qid2negs[query]
        momentum_negative_pool = self.qid2prev_negs[query]

        n_current = int((2 * n) // 3) 
        n_momentum = int(n - n_current)

        current_negatives = random.sample(current_negative_pool, n_current)
        momentum_negatives = random.sample(momentum_negative_pool, n_momentum)

        return current_negatives + momentum_negatives


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--qrels_path', type=str, required=True)
    parser.add_argument('--run_path', type=str, required=True)
    parser.add_argument('--prev_run_path', type=str, required=False)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)

    args = parser.parse_args()

    if args.prev_run_path is None:
        print("sampling from: ")
        print("Current: ", args.run_path)
        sampler = RandomHardNegatives(args.qrels_path, args.run_path, args.out_path)
    else:
        print("sampling from: ")
        print("Current: ", args.run_path)
        print("Momentum: ", args.prev_run_path)
        sampler = RandomHardNegativesWithMomentum(args.qrels_path, args.run_path, args.out_path, args.prev_run_path)

    print(sampler.__class__)
    sampler.set_seed(17121998)
    sampler.sample_hard_negatives(args.n)
