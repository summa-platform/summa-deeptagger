#!/usr/bin/env python3

from __future__ import print_function

import sys, os, json
import numpy as np

# CPU only by default
os.environ['THEANO_FLAGS'] = "optimizer=fast_compile,device=cpu,lib.cnmem=1,floatX=float32"

def enable_gpu():
    # GPU
    os.environ['THEANO_FLAGS'] = "optimizer=fast_compile,device=gpu,lib.cnmem=1,floatX=float32"

# import processing
# from processing import load_word_vectors, clean, extract_wordids
# from classify import DeepTag
from nltk.tokenize import sent_tokenize, word_tokenize

basedir = os.path.dirname(__file__)
# processing.embeddings_folder = os.path.join(basedir, processing.embeddings_folder)

try:
    unicode
except NameError:
    unicode = str

class DeepTagger:

    languages = ["english"] #,"portuguese","german","arabic","persian","russian","ukrainian","spanish"]
    label_types = ['rou','cat','kw']
    labels = {}
    taggers = {}

    @classmethod
    def initialize(cls, init_instances=True):

        import processing
        from processing import load_word_vectors
        processing.embeddings_folder = os.path.join(basedir, processing.embeddings_folder)

        wdim = 512
        cls.wvec, cls.vocab = load_word_vectors(['%s.pkl.gz' % language for language in cls.languages], emb=wdim)
        for language in cls.languages:
            cls.wvec[language] = np.array(cls.wvec[language])

        for language in cls.languages:
            for label_type in cls.label_types:
                with open(os.path.join(basedir, 'models/labels-%s_%s.p' % (label_type, language)), 'r') as f:
                    cls.labels[(language,label_type)] = json.load(f)

        if init_instances:
            for language in cls.languages:
                for label_type in cls.label_types:
                    cls.taggers[(language, label_type)] = cls(language, label_type)

    @classmethod
    def select_one(cls, language, label_type):
        return cls.taggers[(language, label_type)]

    @classmethod
    def select(cls, label_type):
        for language in cls.languages:
            yield cls.taggers[(language, label_type)]

    def __init__(self, language, label_type):
        self.language = language
        self.label_type = label_type

        test_path = os.path.join(basedir, "models/atttd/model-%s_%s.h5" % (label_type, language))
        comp = 'atttd'
        act = 'relu'
        self.wdim = wdim = 512
        gt = 100 if label_type == 'kw' else 0
        swpad = 50
        spad = 50
        sdim = 256
        ddim = 256
        bs = 16
        ep = 20

        self.top_k = 10

        from classify import DeepTag

        self.dt = DeepTag(ep=ep,
                        # wdim=int(wdim.replace('g-','')),
                        wdim=wdim,
                        comp=comp,
                        act=act,
                        swpad=swpad,
                        spad=spad,
                        sdim=sdim,
                        ddim=ddim,
                        ltype=label_type,
                        gt=gt,
                        bs=bs)

        self.labels = self.labels[(language,label_type)]

        self.dt.build_model(self.labels)
        self.dt.model.load_weights(test_path)

    def load(self, docs, lang):

        from processing import clean, extract_wordids

        vocab = self.vocab

        X_idxs, Y_idxs = [],[]
        skipped = []

        for i, doc in enumerate(docs):
            if type(doc) in (str, unicode):
                doc = json.loads(doc)
            if type(doc) is bytes:
                doc = json.loads(doc.decode('utf8'))
            elif not isinstance(doc, dict):
                try:
                    doc = json.loads(doc.read())
                except AttributeError:
                    pass

            if 'text' not in doc:
                skipped.append(i)
                continue

            ltype = self.label_type

            title = doc["name"].lower()
            teaser = doc.get("teaser", "").lower()
            body = doc['text'].lower()
            keywords = []
            for refgroup in doc.get('referenceGroups', []):
                if refgroup["type"] == "Keywords":
                    if lang == "arabic":
                        keywords = [w["name"].strip().lower() for w in refgroup['items']]
                    else:
                        keywords = [w["name"].strip().lower() for w in refgroup['items']]
            category = doc["categoryName"].lower()
            routes = doc["trackingInfo"]["customCriteria"]["x10"].lower().split("::")[2:]
            sentences = [clean(title)]
            sentences += sent_tokenize(clean(teaser))
            sentences += sent_tokenize(clean(body))
            # Exctract word ids and vectors per sentence
            x, x_ids = [], []
            for sentence in sentences:
                vecs_ids = []
                for word in word_tokenize(sentence):
                    try:
                        idx = self.vocab[lang].index(word)
                        vecs_ids.append(idx)
                    except:
                        continue
                if len(vecs_ids) > 0:
                    x_ids.append(vecs_ids)

            if ltype == "kw":
                y_ids = extract_wordids(keywords, lang, vocab)
            elif ltype == "cat":
                y_ids = extract_wordids([category], lang, vocab)
            elif ltype == "rou":
                y_ids = extract_wordids(routes, lang, vocab)

            if not x_ids:
                skipped.append(i)
                continue

            X_idxs.append(x_ids)
            Y_idxs.append(y_ids)

        return X_idxs, Y_idxs, skipped

    def __call__(self, docs, input_language):
        language = self.language
        label_type = self.label_type

        vocab = self.vocab
        top_k = self.top_k
        labels = self.labels

        xt_ids, yt_ids, skipped = self.load(docs, input_language)

        if not xt_ids:
            for i in range(skipped):
                yield None
            return

        satts,watts,scores = self.dt.eval(xt_ids, yt_ids, 0.20, self.wvec[input_language], False, bs=128, av='micro')

        i = 0
        skipped_i = 0

        for rawscores,satts,watts,xt_ids,yt_ids in zip(scores[0],satts,watts,xt_ids,yt_ids):

            while skipped_i < len(skipped) and i == skipped[skipped_i]:
                i += 1
                skipped_i += 1
                yield None

            idxs = np.argsort(rawscores)[::-1]
            scores = np.sort(rawscores)[::-1]
            tags = []
            lang_tags = dict(tags=tags, satts=[], watts=[], text=[], labels=[])
            for i, idx in enumerate(idxs[:top_k]):
                toks = labels[idx].split('_')
                tag = []
                for j, tok in enumerate(toks):
                    tag.append(vocab[language][int(tok)])
                tags.append([' '.join(tag), '%.5f' % scores[i]])
                if len(satts) > 0:
                    sats_text, wats_text = [], []
                    for sidx, sat in enumerate(satts):
                        wat = watts[sidx]
                        cur_text = []
                        sats_text.append("%.3f" % sat)
                        for wval in wat:
                            cur_text.append("%.3f" % wval)
                        wats_text.append(cur_text)
                    lang_tags['satts'] = sats_text
                    lang_tags['watts'] = wats_text
                text, label_text = [], []
                for sentence_ids in xt_ids:
                    text.append([vocab[input_language][idx] for idx in sentence_ids])
                for label_ids in yt_ids:
                    lab = []
                    for idy in label_ids:
                        lab.append(vocab[input_language][idy])
                    label_text.append(' '.join(lab))
                lang_tags['text'] = text
                lang_tags['labels'] = label_text

            i += 1

            yield lang_tags


def rest_service(port=6000, host='0.0.0.0'):
    from bottle import request, Bottle, abort, static_file, response, run, hook
    app = Bottle()

    all_label_types = ','.join(DeepTagger.label_types)
    all_languages = ','.join(DeepTagger.languages)
    default_input_language = 'english'

    @app.post('/tag')
    def tag():
        label_types = list(filter(DeepTagger.label_types.__contains__,
            map(str.strip, request.query.get('label_types', all_label_types).lower().split(','))))
        languages = list(filter(DeepTagger.languages.__contains__,
            map(str.strip, request.query.get('languages', all_languages).lower().split(','))))
        input_language = (lambda x: x in DeepTagger.languages and x)(request.query.get('input_language', default_input_language).lower().strip())
        if not label_types or not languages or not input_language:
            error = []
            if not label_types:
                error.append('invalid argument "languages"')
                print('error: invalid argument "languages"', file=sys.stderr)
            if not label_types:
                error.append('invalid argument "label_types"')
                print('error: invalid argument "label_types"', file=sys.stderr)
            if not input_language:
                error.append('invalid argument "input_language"')
                print('error: invalid argument "input_language"', file=sys.stderr)
            response.status = 400
            return 'error: %s' % ', '.join(error)
        print('label types:', ','.join(label_types), file=sys.stderr)
        print('languages:', ','.join(languages), file=sys.stderr)
        print('input language:', input_language, file=sys.stderr)
        data = request.body.read()
        print('%i bytes of data' % len(data), file=sys.stderr)
        if not data:
            response.status = 400
            return 'error: no data'
        try:
            data = json.loads(data.decode('utf8'))
        except:
            print('error: invalid input data, JSON parser error', file=sys.stderr)
            import traceback
            traceback.print_exc()
            response.status = 400
            return 'error: invalid input data, JSON parser error'
        if isinstance(data, dict):
            docs = [data]
        elif isinstance(data, list):
            docs = data
        else:
            print('error: bad data', file=sys.stderr)
            response.status = 400
            return 'error: bad data'
        single = isinstance(data, dict)
        result = {}
        for label_type in label_types:
            for language in languages:
                tagger = DeepTagger.select_one(language, label_type)
                r = list(tagger(docs, input_language))
                if single:
                    r = r[0] if r else []
                result['%s_%s' % (label_type,language)] = r
        response.status = 200
        return json.dumps(result)

    run(app, host=host, port=port)


if __name__ == "__main__":

    import argparse 
 
    parser = argparse.ArgumentParser(description='DeepTagger', formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('--label-types', '--label-type', '-l', type=str, default=','.join(DeepTagger.label_types),
                                                                help='comma separated list of labels') 
    parser.add_argument('--data-dir', '-d', type=str, default=basedir, help='specify data (models, word vectors) directory') 
    parser.add_argument('--input-language', '-i', type=str, default='english', help='input language') 
    parser.add_argument('--languages', '-L', type=str, default=','.join(DeepTagger.languages), help='comma separated list of languages') 
    parser.add_argument('--rest', '-r', action='store_true', help='start REST API service') 
    parser.add_argument('--port', '-p', type=int, default=6000, help='specify port for REST API service') 
    parser.add_argument('--bind', '-b', type=str, default='0.0.0.0', help='specify listen host for REST API service (0.0.0.0 for any)') 
    parser.add_argument('--rest-help', action='store_true', help='show info about REST API') 
    parser.add_argument('--gpu', action='store_true', help='enable GPU support')
    parser.add_argument('path', metavar='PATH', type=str, nargs='*', help='filename or directory with JSON documents')

    args = parser.parse_args()

    if args.gpu:
        enable_gpu()
    
    basedir = args.data_dir

    if args.rest_help:
        print(file=sys.stderr)
        print('DeepTagger REST API:', file=sys.stderr)
        print(file=sys.stderr)
        print('POST /tag?input_language=english&label_types=%s&languages=%s' % (','.join(DeepTagger.label_types), ','.join(DeepTagger.languages)),
                file=sys.stderr)
        print(file=sys.stderr)
        print('Query arguments given above are defaults and can be omitted.', file=sys.stderr)
        print(file=sys.stderr)
        print('Payload: JSON document')
        print(file=sys.stderr)
        print('Response: JSON document with result tags.')
        print(file=sys.stderr)
        print(file=sys.stderr)
        print('Sample test with curl:', file=sys.stderr)
        print('$ curl -X POST -d @document.json http://localhost:6000/tag?label_types=kw,cat&input_language=english', file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(0)

    if not args.rest and not args.path:
        print('error: either --rest or path must be specified', file=sys.stderr)
        sys.exit(1)

    if args.input_language not in DeepTagger.languages:
        print('error: unknown input language:', args.input_language, file=sys.stderr)
        sys.exit(1)

    input_language = args.input_language
    label_types = [label_type for label_type in args.label_types.split(',') if label_type in DeepTagger.label_types]

    DeepTagger.languages = languages = [language for language in args.languages.split(',') if language in DeepTagger.languages]

    if input_language not in languages:
        DeepTagger.languages.append(input_language)

    if not languages:
        print('error: no valid languages selected', file=sys.stderr)
        sys.exit(1)

    print('Languages:', ','.join(languages), file=sys.stderr)
    print('Input Language:', input_language, file=sys.stderr)
    print('Label Types:', ','.join(label_types), file=sys.stderr)

    # redirect stdout to stderr
    # http://stackoverflow.com/questions/6796492/temporarily-redirect-stdout-stderr/22434728#22434728
    # http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    from contextlib import contextmanager
    @contextmanager
    def redirect_stdout(to=sys.stderr):
        sys.stdout.flush()
        stdout_fd = sys.stdout.fileno()
        with os.fdopen(os.dup(stdout_fd), 'wb') as dup:
            try:
                os.dup2(to.fileno(), stdout_fd)
                # yield sys.stdout
                yield dup
            finally:
                sys.stdout.flush()
                os.dup2(dup.fileno(), stdout_fd)


    # debug:
    # label_type = 'kw'
    # input_language = 'english'
    # languages = DeepTagger.languages = ['english']
    # label_types = DeepTagger.label_types = ['kw']


    print('Initializing...', file=sys.stderr)
    with redirect_stdout():
        DeepTagger.initialize()
    print('Initialized', file=sys.stderr)


    if args.rest:
        rest_service(port=args.port, host=args.bind)
        sys.exit(0)
    

    def walk(path, ext='.json'):
        if path.endswith('.json') and os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for basedir, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    if filename.lower().endswith(ext):
                        yield os.path.join(basedir, filename)

    def get_filelist(paths):
        for path in paths:
            for fn in walk(path):
                yield fn

    def open_files(filelist):
        for fn in filelist:
            with open(fn) as f:
                yield f

    filelist = list(get_filelist(args.path))

    with redirect_stdout() as out:
        first = True
        out.write('[')
        for label_type in label_types:
            for language in languages:
                out.write('%s{"language":%s,"label_type":%s,"documents":{\n' % ('' if first else ',\n', json.dumps(language), json.dumps(label_type)))
                print('Label Type: %s ; Language: %s' % (label_type, language), file=sys.stderr)
                tagger = DeepTagger.select_one(language, label_type)
                first_file = True
                for tags,fn in zip(tagger(open_files(filelist), input_language), filelist):
                    out.write('' if first_file else ',\n')
                    print('File:', fn, file=sys.stderr)
                    out.write('%s:' % json.dumps(fn))
                    json.dump(tags, out)
                    if first_file:
                        first_file = False
                out.write('}}')
                if first:
                    first = False
        out.write(']\n')

    # for path in args.path:
    #     for fn in walk(path):
    #         print('File:', fn, file=sys.stderr)
    #         for label_type in label_types:
    #             for language in languages:
    #                 tagger = DeepTagger.select_one(language, label_type)
    #                 with open(fn) as f:
    #                     tags = next(tagger([f], input_language))
    #                 json.dump(tags, sys.stdout)
    #             print(tagger.label_type, tagger.language, lang_tags)
    #
    # documentsdir = 'documents/%s' % input_language
    # for fn in os.listdir(documentsdir):
    #     if not fn.endswith('.json'):
    #         continue
    #     fn = os.path.join(documentsdir, fn)
    #     if not os.path.isfile(fn):
    #         continue
    #     print(fn)
    #     for tagger in DeepTagger.select(label_type):
    #         print(tagger.language, tagger.label_type)
    #         with open(fn) as f:
    #             lang_tags = next(tagger([f], input_language))
    #         print(tagger.label_type, tagger.language, lang_tags)

	# json.dump(lang_tags, open('documents/%s/tags/%s_%s.json' % (input_language, label_type,language) , 'w'))
