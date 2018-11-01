#!/usr/bin/env python3

import sys, os, asyncio, traceback, time, json

from worker_pool import WorkerProcessPool, ErrorMessage

from deeptagger import DeepTagger, enable_gpu
import deeptagger


class RejectError(Exception): pass
class RejectRequeueError(Exception): pass
class ErrorMessage(Exception): pass
class NoReply(Exception): pass


name = 'SUMMA-DT'      # required by rabbitmq module

MODEL_DIR = os.path.join(os.path.dirname(__file__), "data")
debug = False


def init(args=None):
    global pool, debug
    log('Initialize DeepTagger ...')
    if args.gpu:
        enable_gpu()
    debug = args.debug
    deeptagger.basedir = args.model_dir
    DeepTagger.initialize()
    log('DeepTagger initialized!')
    pool = WorkerProcessPool(worker_run, init_module, count=args.PARALLEL, heartbeat_pause=args.heartbeat_pause, init_args=(args,))
    pool.start()
    # give some time for workers to start
    # time.sleep(5)
    pool.watch_heartbeats(args.restart_timeout, args.refresh, args.max_retries_per_job)


def setup_argparser(parser):
    env = os.environ
    parser.add_argument('--model-dir', type=str, default=env.get('MODEL_DIR', MODEL_DIR), help='model directory (env MODEL_DIR)')
    parser.add_argument('--gpu', action='store_true', help='enable GPU mode') 
    parser.add_argument('--heartbeat-pause', type=int, default=env.get('HEARTBEAT_PAUSE', 10),
            help='pause in seconds between heartbeats (or set env variable HEARTBEAT_PAUSE)')
    parser.add_argument('--refresh', type=int, default=env.get('REFRESH', 5),
            help='seconds between pulse checks (or set env variable REFRESH)')
    parser.add_argument('--restart-timeout', type=int, default=env.get('RESTART_TIMEOUT', 5*60),
            help='max allowed seconds between heartbeats, will restart worker if exceeded (or set env variable RESTART_TIMEOUT)')
    parser.add_argument('--max-retries-per-job', type=int, default=env.get('MAX_RETRIES_PER_JOB', 3),
            help='maximum retries per job (or set env variable MAX_RETRIES_PER_JOB)')


def shutdown():
    global pool
    return pool.terminate()


def reset():
    global pool
    pool.reset()


async def process_message(task_data, loop=None, send_reply=None, metadata=None, reject=None, **kwargs):
    global pool
    async with pool.acquire() as worker:
        result = await worker((task_data, 'english', ['english'], DeepTagger.label_types), send_reply)
        if debug:
            print(result)
        return result


# --- private ---

async def worker_run(task, partial_result_callback=None, loop=None, heartbeat=None, *args, **kwargs):
    documents, input_language, languages, label_types = task
    single = isinstance(documents, dict)
    if single:
        documents = [documents]
    result = {}
    for label_type in label_types:
        for language in languages:
            tagger = DeepTagger.select_one(language, label_type)
            r = list(tagger(documents, input_language))
            heartbeat()
            if single:
                r = r[0] if r else []
            result['%s_%s' % (label_type,language)] = r
    return result


def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def init_module(args):
    pass



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Deep Tagger Task', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', action='store_true', help='enable debug mode') 
    parser.add_argument('--parallel', '-n', dest='PARALLEL', metavar='PORT', type=int, default=os.environ.get('PARALLEL',1),
            help='messages to process in parallel (or set env variable PARALLEL)')
    parser.add_argument('filename', type=str, default='test.json', nargs='?', help='JSON file with task data')

    setup_argparser(parser)

    args = parser.parse_args()

    init(args)

    print('Reading', args.filename)
    with open(args.filename, 'r') as f:
        task_data = json.load(f)
    metadata = {}

    async def print_partial(partial_result):
        print('Partial result:')
        print(partial_result)

    try:
        loop = asyncio.get_event_loop()
        # loop.set_debug(True)
        result = loop.run_until_complete(process_message(task_data, loop=loop, send_reply=print_partial, metadata=metadata))
        print('Result:')
        print(result)
    except KeyboardInterrupt:
        print('INTERRUPTED')
    except:
        print('EXCEPTION')
        traceback.print_exc()
        # raise
    finally:
        shutdown()
