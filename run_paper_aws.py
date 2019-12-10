# from run_basic_v2 import *
import boto3
import json
import sys
import logging
import re
import shutil
import argparse
import paramiko
import numpy as np
import time
import os
from collections import OrderedDict
import asyncio
import random
import pandas as pd
import datetime

'''
This is a script to distribute experiments in AWS.

It is not in a state where it can be used off the shelf without
a few modifications. It is meant as a tool to replicate the paper.

'''

def get_start_script(script='experiment_runner.sh'):
    '''
    Reads the specified script as a string and returns.
    '''
    with open(script, 'r') as f:
        return f.read()

def start_instance(client, instance_name, key_name, security_group, instance_type, image_id, VPCid, script_to_run):
    '''
    Starts instances using an ec2 resource.
    '''
    # print(start_script)
    instances = client.create_instances(
                ImageId=image_id,
                MinCount=1,
                MaxCount=1,
                KeyName=key_name,
                SecurityGroupIds=[security_group],
                InstanceType=instance_type,
                # InstanceInitiatedShutdownBehavior='terminate',
                SubnetId=VPCid,
                UserData=script_to_run,)
    client.create_tags(Resources=[instances[0].id],
                       Tags=[{'Key':'Name','Value': instance_name}])
    return instances

def read_s3_bucket(s3, bucket_name):
    '''
    Downloads most recent models from s3
    '''

    bucket = s3.Bucket(bucket_name)

    experiments = [int(x.key.split('/')[0].split('_')[1]) for x in bucket.objects.all()]

    return experiments

def remove_directory(args):
    [os.remove(os.path.join(os.getcwd(), args.city,x)) for x in os.listdir(os.path.join(os.getcwd(), args.city))]
    os.rmdir(os.path.join(os.getcwd(), args.city))


def create_experiment_metadata(exp_type):

    np.random.seed(0)

    if exp_type == 'toy_graph':
        hyperparams = {}
        count = 0
        # MDP
        # for trajectory_length in [8, 16, 32, 64, 128, 256, 512, 1024]:
        #     for horizon in [4, 16]:
        #         for pi0 in [1]:
        #             for pi1 in [4]:
        #                 for stochastic_reward in [0]:
        #                     for stochastic_env in [1]:
        #                         for sparse_reward in [0]:
        #                             for _ in range(100): # pick 10 different seeds
        #                                 hyperparams[count] = OrderedDict({'gamma': .98,
        #                                                       'horizon': horizon,
        #                                                       'base_policy': pi0,
        #                                                       'eval_policy': pi1,
        #                                                       'stochastic_env': stochastic_env,
        #                                                       'stochastic_rewards': stochastic_reward,
        #                                                       'sparse_rewards': sparse_reward,
        #                                                       'num_traj': trajectory_length,
        #                                                       'is_pomdp': 0,
        #                                                       'pomdp_horizon': 2,
        #                                                       'seed': np.random.randint(0, 2**16),
        #                                                       'modeltype': 'tabular',
        #                                                       'to_regress_pi_b': 0,
        #                                                       })
        #                                 count += 1
        for trajectory_length in [256, 512, 1024]:
            for horizon in [4, 16]:
                for pi0 in [1, 3]:
                    for pi1 in [4]:
                        for stochastic_reward in [0, 1]:
                            for stochastic_env in [0, 1]:
                                for sparse_reward in [0, 1]:
                                    for _ in range(10): # pick 10 different seeds
                                        hyperparams[count] = OrderedDict({'gamma': .98,
                                                              'horizon': horizon,
                                                              'base_policy': pi0,
                                                              'eval_policy': pi1,
                                                              'stochastic_env': stochastic_env,
                                                              'stochastic_rewards': stochastic_reward,
                                                              'sparse_rewards': sparse_reward,
                                                              'num_traj': trajectory_length,
                                                              'is_pomdp': 0,
                                                              'pomdp_horizon': 2,
                                                              'seed': np.random.randint(0, 2**16),
                                                              'modeltype': 'tabular',
                                                              'to_regress_pi_b': 0,
                                                              })
                                        count += 1
        # POMDP
        for trajectory_length in [256, 512, 1024]:
            for horizon in [2, 16]:
                for pi0 in [1, 3]:
                    for pi1 in [4]:
                        for stochastic_reward in [0, 1]:
                            for stochastic_env in [0, 1]:
                                for sparse_reward in [0, 1]:
                                    for _ in range(10): # pick 10 different seeds
                                        hyperparams[count] = OrderedDict({'gamma': .98,
                                                              'horizon': horizon,
                                                              'base_policy': pi0,
                                                              'eval_policy': pi1,
                                                              'stochastic_env': stochastic_env,
                                                              'stochastic_rewards': stochastic_reward,
                                                              'sparse_rewards': sparse_reward,
                                                              'num_traj': trajectory_length,
                                                              'is_pomdp': 1,
                                                              'pomdp_horizon': 6 if horizon == 16 else 2,
                                                              'seed': np.random.randint(0, 2**16),
                                                              'modeltype': 'tabular',
                                                              'to_regress_pi_b': 0,
                                                              })
                                        count += 1
        # LOW data regime
        for trajectory_length in [8, 16, 32, 64, 128]:
            for horizon in [4, 16]:
                for pi0 in [1, 3]:
                    for pi1 in [4]:
                        for stochastic_reward in [0, 1]:
                            for stochastic_env in [0, 1]:
                                for sparse_reward in [0, 1]:
                                    for _ in range(10): # pick 10 different seeds
                                        hyperparams[count] = OrderedDict({'gamma': .98,
                                                              'horizon': horizon,
                                                              'base_policy': pi0,
                                                              'eval_policy': pi1,
                                                              'stochastic_env': stochastic_env,
                                                              'stochastic_rewards': stochastic_reward,
                                                              'sparse_rewards': sparse_reward,
                                                              'num_traj': trajectory_length,
                                                              'is_pomdp': 0,
                                                              'pomdp_horizon': 2,
                                                              'seed': np.random.randint(0, 2**16),
                                                              'modeltype': 'tabular',
                                                              'to_regress_pi_b': 0,
                                                              })
                                        count += 1
        return hyperparams
    elif exp_type == 'gridworld':
        hyperparams = {}
        count = 0
        for trajectory_length in [64, 128, 256, 512, 1024]:
            for horizon in [25]:
                for pi0,pi1 in zip([100, 80, 60, 40, 20], [10, 10, 10, 10, 10]):
                        for stochastic_reward in [0]:
                            for stochastic_env in [0]:
                                for sparse_reward in [0]:
                                    for _ in range(10): # pick 10 different seeds
                                        hyperparams[count] = OrderedDict({'gamma': .96,
                                                              'horizon': horizon,
                                                              'base_policy': pi0,
                                                              'eval_policy': pi1,
                                                              'stochastic_env': stochastic_env,
                                                              'stochastic_rewards': stochastic_reward,
                                                              'sparse_rewards': sparse_reward,
                                                              'num_traj': trajectory_length,
                                                              'is_pomdp': 0,
                                                              'pomdp_horizon': 2,
                                                              'seed': np.random.randint(0, 2**16),
                                                              'modeltype': 'tabular',
                                                              'to_regress_pi_b': 1,
                                                              })
                                        count += 1
        return hyperparams
    elif exp_type == 'pixel_gridworld':
        hyperparams = {}
        count = 0
        for trajectory_length in [64, 128, 256, 512]:
            for horizon in [25]:
                for pi0,pi1 in zip([100, 80, 60, 40, 20], [10, 10, 10, 10, 10]):
                        for stochastic_reward in [0]:
                            for stochastic_env in [0]:
                                for sparse_reward in [0]:
                                    for _ in range(10): # pick 10 different seeds
                                        hyperparams[count] = OrderedDict({'gamma': .96,
                                                              'horizon': horizon,
                                                              'base_policy': pi0,
                                                              'eval_policy': pi1,
                                                              'stochastic_env': stochastic_env,
                                                              'stochastic_rewards': stochastic_reward,
                                                              'sparse_rewards': sparse_reward,
                                                              'num_traj': trajectory_length,
                                                              'is_pomdp': 0,
                                                              'pomdp_horizon': 2,
                                                              'seed': np.random.randint(0, 2**16),
                                                              'modeltype': 'conv',
                                                              'to_regress_pi_b': 0,
                                                              })
                                        count += 1
        for trajectory_length in [64, 128, 256]:
            for horizon in [25]:
                for pi0,pi1 in zip([100, 80, 60, 40, 20], [10, 10, 10, 10, 10]):
                        for stochastic_reward in [0]:
                            for stochastic_env in [0]:
                                for sparse_reward in [0]:
                                    for _ in range(10): # pick 10 different seeds
                                        hyperparams[count] = OrderedDict({'gamma': .96,
                                                              'horizon': horizon,
                                                              'base_policy': pi0,
                                                              'eval_policy': pi1,
                                                              'stochastic_env': stochastic_env,
                                                              'stochastic_rewards': stochastic_reward,
                                                              'sparse_rewards': sparse_reward,
                                                              'num_traj': trajectory_length,
                                                              'is_pomdp': 0,
                                                              'pomdp_horizon': 2,
                                                              'seed': np.random.randint(0, 2**16),
                                                              'modeltype': 'conv',
                                                              'to_regress_pi_b': 1,
                                                              })
                                        count += 1
        for trajectory_length in [64, 128, 256]:
            for horizon in [25]:
                for pi0,pi1 in zip([100, 80, 60, 40, 20], [10, 10, 10, 10, 10]):
                        for stochastic_reward in [0]:
                            for stochastic_env in [1]:
                                for sparse_reward in [0]:
                                    for _ in range(10): # pick 10 different seeds
                                        hyperparams[count] = OrderedDict({'gamma': .96,
                                                              'horizon': horizon,
                                                              'base_policy': pi0,
                                                              'eval_policy': pi1,
                                                              'stochastic_env': stochastic_env,
                                                              'stochastic_rewards': stochastic_reward,
                                                              'sparse_rewards': sparse_reward,
                                                              'num_traj': trajectory_length,
                                                              'is_pomdp': 0,
                                                              'pomdp_horizon': 2,
                                                              'seed': np.random.randint(0, 2**16),
                                                              'modeltype': 'conv',
                                                              'to_regress_pi_b': 1,
                                                              })
                                        count += 1
        return hyperparams
    elif exp_type == 'toy_mc':
        hyperparams = {}
        count = 0
        for trajectory_length in [128, 256, 512, 1024]:
            for horizon in [250]:
                for pi0,pi1 in zip([45,60,45,60,80,20], [45,60,60,45,20,80]):
                        for stochastic_reward in [0]:
                            for stochastic_env in [0]:
                                for sparse_reward in [0]:
                                    for _ in range(10): # pick 10 different seeds
                                        hyperparams[count] = OrderedDict({'gamma': .99,
                                                              'horizon': horizon,
                                                              'base_policy': pi0,
                                                              'eval_policy': pi1,
                                                              'stochastic_env': stochastic_env,
                                                              'stochastic_rewards': stochastic_reward,
                                                              'sparse_rewards': sparse_reward,
                                                              'num_traj': trajectory_length,
                                                              'is_pomdp': 0,
                                                              'pomdp_horizon': 2,
                                                              'seed': np.random.randint(0, 2**16),
                                                              'modeltype': 'tabular',
                                                              'to_regress_pi_b': 0,
                                                              })
                                        count += 1
        return hyperparams
    elif exp_type == 'mc':
        hyperparams = {}
        count = 0
        for trajectory_length in [128, 256, 512]:
            for horizon in [250]:
                for pi0,pi1 in zip([1,3,3,1],[0,0,1,3]):
                        for modeltype in ['linear', 'mlp']:
                            for stochastic_reward in [0]:
                                for stochastic_env in [0]:
                                    for sparse_reward in [0]:
                                        for _ in range(10): # pick 10 different seeds
                                            hyperparams[count] = OrderedDict({'gamma': .99,
                                                                  'horizon': horizon,
                                                                  'base_policy': pi0,
                                                                  'eval_policy': pi1,
                                                                  'stochastic_env': stochastic_env,
                                                                  'stochastic_rewards': stochastic_reward,
                                                                  'sparse_rewards': sparse_reward,
                                                                  'num_traj': trajectory_length,
                                                                  'is_pomdp': 0,
                                                                  'pomdp_horizon': 2,
                                                                  'seed': np.random.randint(0, 2**16),
                                                                  'modeltype': modeltype,
                                                                  'to_regress_pi_b': 0,
                                                                  })
                                            count += 1
        return hyperparams
    elif exp_type == 'pixel_mc':
        hyperparams = {}
        count = 0
        for trajectory_length in [512]:
            for horizon in [500]:
                for pi0,pi1 in zip([2,1,2],[0,0,1]):
                        for modeltype in ['conv']:
                            for stochastic_reward in [0]:
                                for stochastic_env in [0]:
                                    for sparse_reward in [0]:
                                        for _ in range(10): # pick 10 different seeds
                                            hyperparams[count] = OrderedDict({'gamma': .97,
                                                                  'horizon': horizon,
                                                                  'base_policy': pi0,
                                                                  'eval_policy': pi1,
                                                                  'stochastic_env': stochastic_env,
                                                                  'stochastic_rewards': stochastic_reward,
                                                                  'sparse_rewards': sparse_reward,
                                                                  'num_traj': trajectory_length,
                                                                  'is_pomdp': 0,
                                                                  'pomdp_horizon': 2,
                                                                  'seed': np.random.randint(0, 2**16),
                                                                  'modeltype': modeltype,
                                                                  'to_regress_pi_b': 0,
                                                                  })
                                            count += 1
        return hyperparams
    elif exp_type == 'enduro':
        hyperparams = {}
        count = 0
        np.random.seed(1)
        for trajectory_length in [512]:
            for horizon in [500]:
                for pi0,pi1 in zip([2,1,2],[0,0,1]):
                        for modeltype in ['conv']:
                            for stochastic_reward in [0]:
                                for stochastic_env in [0]:
                                    for sparse_reward in [0]:
                                        for _ in range(10): # pick 10 different seeds
                                            hyperparams[count] = OrderedDict({'gamma': .9999,
                                                                  'horizon': horizon,
                                                                  'base_policy': pi0,
                                                                  'eval_policy': pi1,
                                                                  'stochastic_env': stochastic_env,
                                                                  'stochastic_rewards': stochastic_reward,
                                                                  'sparse_rewards': sparse_reward,
                                                                  'num_traj': trajectory_length,
                                                                  'is_pomdp': 0,
                                                                  'pomdp_horizon': 2,
                                                                  'seed': np.random.randint(0, 2**16),
                                                                  'modeltype': modeltype,
                                                                  'to_regress_pi_b': 0,
                                                                  })
                                            count += 1
        return hyperparams
    else:
        raise NotImplemented

class TaskRunner(object):
    def __init__(self, sessions, hyperparams, args):
        self.complete = False

        self.sessions = sessions
        self.hyperparams = hyperparams
        self.args = args
        self.env_type = '-'.join(args.env_type.split('_'))
        self.terminate_on_end = self.env_type not in ['pixel-mc', 'enduro']
        self.start_instances()

    def start_instances(self):
        startup_script = get_start_script('experiment_runner.sh')
        self.info = {}

        if self.env_type in ['toy-graph', 'toy-mc', 'mc', 'gridworld', 'pixel-gridworld']:
            #use cpus
            instances = []
            count = 0
            for _ in range(self.args.number_of_instances):
                try:
                    region = 'us-east-2'
                    instance_name = '-'.join(['OPE', 'minion-%s' % count])

                    # These need to be changed to *your* AWS account
                    instances.append(start_instance(self.sessions[region]['ec2'], instance_name, 'username', 'sg-aaaaaa', 't2.medium', 'ami-aaaaaaaaaaaaaaaa', 'subnet-aaaaaaaa', startup_script))

                    self.info[instances[-1][0]] = {}
                    self.info[instances[-1][0]]['username'] = 'ubuntu'
                    self.info[instances[-1][0]]['pem_addr'] = os.path.join('your.pem') # folder path to aws instance key
                    self.info[instances[-1][0]]['aws_region'] = region
                    count += 1
                except:
                    pass
        elif self.env_type in ['pixel-mc', 'enduro']:
            #use 3 gpus
            self.args.number_of_instances = 3

            # TODO: This should be changed to your gpu instance or made automatic
            instances = [[self.sessions['us-east-2']['ec2'].Instance('i-aaaaaaaaaaaaaaaaa')],
                         [self.sessions['us-west-2']['ec2'].Instance('i-aaaaaaaaaaaaaaaaa')],
                         [self.sessions['us-east-1']['ec2'].Instance('i-aaaaaaaaaaaaaaaaa')]]

            self.info[instances[0][0]] = {}
            self.info[instances[0][0]]['username'] = 'ubuntu'
            self.info[instances[0][0]]['pem_addr'] = os.path.join('your.pem') # folder path to aws instance key
            self.info[instances[0][0]]['aws_region'] = 'us-east-2'

            self.info[instances[1][0]] = {}
            self.info[instances[1][0]]['username'] = 'ubuntu'
            self.info[instances[1][0]]['pem_addr'] = os.path.join('your.pem') # folder path to aws instance key
            self.info[instances[1][0]]['aws_region'] = 'us-west-2'

            self.info[instances[2][0]] = {}
            self.info[instances[2][0]]['username'] = 'ubuntu'
            self.info[instances[2][0]]['pem_addr'] = os.path.join('your.pem') # folder path to aws instance key
            self.info[instances[2][0]]['aws_region'] = 'us-east-1'

        else:
            raise NotImplemented


        print('Able to create %s out of (desired) %s instances' % (len(instances), self.args.number_of_instances))


        instances = [x[0] for x in instances]
        self.instances = instances
        self.sshs = {}
        self.tracker = {instance:{'start_time':0, 'current_task':None, 'in_queue':False, 'ready': False, 'attempt':0, 'completed':0} for instance in instances}
        if self.env_type in ['toy-graph', 'toy-mc', 'mc']:
            time.sleep(2*60)


    def print_stats(self):
        df = pd.DataFrame(self.tracker).T
        df['run_time'] = time.time() - df['start_time']
        print(df[['current_task', 'run_time', 'attempt', 'completed']][df['current_task'] >= 0])

    def run(self):
        try:
            self.loop = asyncio.get_event_loop()
            self.available_workers = asyncio.Queue(loop=self.loop)
            producer_coro = self.produce()
            consumer_coro = self.consume()
            self.starttime = time.time()
            self.loop.run_until_complete(asyncio.gather(producer_coro, consumer_coro))
            self.loop.close()
        finally:
            self.breakdown()

    def experiment_done(self, current_task):
        try:
            # TODO: no hardcode aws_region
            self.sessions['us-east-2']['s3'].Object('%s-ope-experiments' % self.env_type, '/'.join(['exp_%05d' % current_task,'results.json']) ).load()
            return True
        except:
            return False

    def is_available(self, worker):
        current_task = self.tracker[worker]['current_task']
        if not self.tracker[worker]['in_queue']:

            if not self.tracker[worker]['ready']:
                client = self.sessions[self.info[worker]['aws_region']]['client']
                statuses = client.describe_instance_status(InstanceIds=[worker.id])

                try:
                    status = statuses['InstanceStatuses'][0]
                    if status['InstanceStatus']['Status'] == 'ok' \
                            and status['SystemStatus']['Status'] == 'ok':
                        self.tracker[worker]['ready'] = True
                    if not self.tracker[worker]['ready']:
                        return False
                    else:
                        worker.load()
                except:
                    return False

            if current_task is None:
                return True
            else:
                if self.experiment_done(current_task):
                    self.tracker[worker]['current_task'] = None
                    self.tracker[worker]['completed'] += 1
                    return True
                else:
                    # Check if hanging
                    if (time.time() - self.tracker[worker]['start_time']) > 180:
                        ssh = self.sshs[worker][0]
                        cmd_to_run='pgrep -a python'
                        stdin4, stdout4, stderr4 = ssh.exec_command(cmd_to_run,timeout=None, get_pty=False)
                        output = ''
                        for line in stdout4: output += line
                        if output == '':
                            hanging = True
                        else:
                            hanging = False

                        #check again. just to make sure
                        if hanging and not self.experiment_done(current_task): # restart experiment

                            self.run_next_experiment(worker, self.tracker[worker]['current_task'])
                            self.tracker[worker]['attempt'] += 1
                            self.tracker[worker]['start_time'] = time.time()

                            # if self.tracker[worker]['attempt'] == 1:
                            #     stdin, stdout, stderr = self.sshs[worker][1:]
                            #     output = ''
                            #     for line in stdout.readlines(): output += line
                            #     print(output)
                            #     output = ''
                            #     for line in stderr.readlines(): output += line
                            #     print(output)

                            if self.tracker[worker]['attempt'] == 3:
                                print('Failed too many attempts')
                                self.breakdown() # Failure
                            return False
                        else:
                            return False
                    else: # Not hanging. Just not done
                        return False

        else:
            return False

    async def produce(self):
        while not (self.is_complete() and all([self.tracker[i]['in_queue'] for i in self.instances])):
            for worker in self.instances:
                await asyncio.sleep(.01)

                if self.is_available(worker):
                    await self.available_workers.put(worker)
                    self.tracker[worker]['in_queue'] = True
            print('*'*40)
            self.print_stats()
            time_elapsed = time.time() - self.starttime
            total_complete = sum([self.tracker[i]['completed'] for i in self.instances])
            time_per = time_elapsed/total_complete if total_complete > 0 else 120
            print('Time Elapsed: {0}. Total Completed: {1}. Time/Experiment: {2}'.format( str(datetime.timedelta(seconds=(time_elapsed))) ,
                                                                                            total_complete,
                                                                                          str(datetime.timedelta(seconds=(time_per))) ,
                                                                                    ))
            await asyncio.sleep(3)

    def generator(self):
        # TODO: No hardcode aws_region
        experiments_already_ran = set(read_s3_bucket(self.sessions['us-east-2']['s3'], '%s-ope-experiments' % self.env_type))
        all_experiments = set(range(len(self.hyperparams)))
        experiments_to_run = all_experiments - experiments_already_ran
        for i in experiments_to_run:
            yield i
        yield None

    def is_complete(self):
        return self.complete

    async def consume(self):
        gen = self.generator()
        while True:

            # wait for an available worker
            worker = await self.available_workers.get()
            exp_num = next(gen)
            if exp_num is None:
                break

            # process the item
            self.tracker[worker]['current_task'] = exp_num
            self.tracker[worker]['start_time'] = time.time()
            self.tracker[worker]['attempt'] = 0
            self.run_next_experiment(worker, exp_num)
            self.tracker[worker]['in_queue'] = False


        self.complete = True

    def run_next_experiment(self, instance, exp_num):

        if instance not in self.sshs:
            username = self.info[instance]['username']
            pem_addr = self.info[instance]['pem_addr']
            aws_region = self.info[instance]['aws_region']

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            privkey = paramiko.RSAKey.from_private_key_file(pem_addr)
            failed = 0

            print(instance.public_dns_name)
            ssh.connect(instance.public_dns_name, username=username, pkey=privkey)
            self.sshs[instance] = [ssh]
        else:
            ssh = self.sshs[instance][0]
        #you can seperate two shell commands by && or ;

        # Make sure that cuda is correct
        add_export = 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}; '
        main_command = 'python3 paper.py distributed %s' % '_'.join(self.env_type.split('-'))

        if self.env_type in ['enduro', 'pixel-mc']:
            cmd_to_run= 'source activate tensorflow_p36; cd fqe; ' + main_command + ' {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12}'.format(*self.hyperparams[exp_num].values())
        else:
            cmd_to_run= 'cd fqe; ' + add_export + main_command + ' {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12}'.format(*self.hyperparams[exp_num].values())
        cmd_to_run = cmd_to_run + ' {0} {1} {2} &> output.txt'.format(exp_num, self.args.access, self.args.secret)
        stdin4, stdout4, stderr4 = ssh.exec_command(cmd_to_run,timeout=None, get_pty=False)
        self.sshs[instance] = [self.sshs[instance][0]] + [stdin4, stdout4, stderr4]

    def breakdown(self):
        time.sleep(5)
        for key, ssh in self.sshs.items():
            ssh[0].close()

        for instance in self.instances:
            if self.terminate_on_end:
                instance.terminate()

def distribute_experiments(args):
    '''
        Splits the task into N number of parallel data creation tasks. Uploads data files to S3.
    '''
    sessions = {}
    for region in ['us-west-2', 'us-east-2', 'us-east-1']:
        sessions[region] = {}
        sessions[region]['session'] = boto3.Session(aws_access_key_id=args.access, aws_secret_access_key=args.secret, region_name=region)
        sessions[region]['s3'] = sessions[region]['session'].resource('s3')
        sessions[region]['ec2'] = sessions[region]['session'].resource('ec2')
        sessions[region]['client'] = sessions[region]['session'].client('ec2')


    hyperparams = create_experiment_metadata(args.env_type)
    with open('%s_experiment_hyperparam.json' % args.env_type, 'w') as outfile:
        json.dump(hyperparams, outfile)

    runner = TaskRunner(sessions, hyperparams, args)
    runner.run()

if __name__ == '__main__':
    print('start')
    parser = argparse.ArgumentParser(description='Distribute experiments across ec2 instances.')

    parser.add_argument('env_type', help='type: toy_graph, toy_mc, mc, pixel_mc, enduro,...')
    parser.add_argument('access', help='AWS Access Key')
    parser.add_argument('secret', help='AWS Secret Access Key')

    parser.add_argument('-N', dest='number_of_instances', default=15, required=False, type=int)


    args = parser.parse_args()

    distribute_experiments(args)
