# -*- coding: utf-8 -*-

import onmt
import onmt.Markdown
import onmt.IO
import argparse
import torch
import codecs

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-src_seq_length', type=int, default=35,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=20,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def makeVocabulary(filename, size):
    "Construct the word and feature vocabs."
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    with codecs.open(filename, "r", "utf-8") as f:
        for sent in f.readlines():
            words = onmt.IO.extractFeatures(sent.split())

            for i in range(len(words)):
                vocab.add(words[i])

    originalSize = vocab.size()
    if size != 0:
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))
    else:
        print('Created dictionary of size %d' % (vocab.size()))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):
    """If `vocabFile` exists, read it in,
    Else, generate from data."""
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)
        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    alignments = []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = codecs.open(srcFile, "r", "utf-8")
    tgtF = codecs.open(tgtFile, "r", "utf-8")

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        srcLine = sline.split()
        tgtLine = tline.split()

        if len(srcLine) <= opt.src_seq_length \
           and len(tgtLine) <= opt.tgt_seq_length:

            # Check truncation condition.
            if opt.src_seq_length_trunc != 0:
                srcLine = srcLine[:opt.src_seq_length_trunc]

            if opt.tgt_seq_length_trunc != 0:
                tgtLine = tgtLine[:opt.tgt_seq_length_trunc]

            srcWords, srcData = onmt.IO.readSrcLine(srcLine, srcDicts)
            src += [srcData]

            tgtWords, tgtData = onmt.IO.readTgtLine(tgtLine, tgtDicts)
            tgt += [tgtData]

            alignments += [onmt.IO.align(srcWords, tgtWords)]
            sizes += [len(srcData)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        alignments = [alignments[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    alignments = [alignments[idx] for idx in perm]

    print(('Prepared %d sentences ' +
          '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, opt.src_seq_length, opt.tgt_seq_length))

    return src, tgt, alignments


def main():

    dicts = {}
    dicts['src'] = onmt.Dict()
    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                           opt.src_vocab_size)

    dicts['tgt'] = initVocabulary('target',
                       opt.train_tgt,
                       opt.tgt_vocab,
                       opt.tgt_vocab_size)

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'], train['alignments'] = makeData(opt.train_src, opt.train_tgt, dicts['src'], dicts['tgt'])
    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'], valid['alignments']  = makeData(opt.valid_src, opt.valid_tgt, dicts['src'], dicts['tgt'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()