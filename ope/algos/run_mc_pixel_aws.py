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
import aiobotocore
import datetime


def get_start_script(script='agg_mini_instance.sh'):
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


def create_experiment_metadata():

    np.random.seed(0)

    # Exper 1: MDP, Graph
    hyperparams = {}
    count = 0
    for num_datapoints in [2048]:
        # for pi0 in range(2):
        #     for pi1 in range(2):
        #         for _ in range(10): # pick 10 different seeds
        hyperparams[count] = OrderedDict({'gamma': .98,
                                          'base_policy': 1,
                                          'eval_policy': 0,
                                          'num_traj': num_datapoints,
                                          'seed': np.random.randint(0, 2**32),
                                          })
                    
        count += 1
    return hyperparams


class TaskRunner(object):
    def __init__(self, client, ec2, s3, hyperparams, args):
        self.complete = False
        self.client = client
        # self.instances = instances
        # self.tracker = {instance:{'start_time':0, 'current_task':None, 'in_queue':False, 'ready': False, 'attempt':0} for instance in instances}
        self.ec2 = ec2
        self.s3 = s3
        self.hyperparams = hyperparams
        self.args = args
        self.start_instances()

    def start_instances(self):
        startup_script = get_start_script('experiment_runner.sh')
    
        instances = []
        count = 0
        # for _ in range(self.args.number_of_instances):
        #     try:
        #         instance_name = '-'.join(['CameronVoloshin-OPE', 'minion-%s' % count])
        #         instances.append(start_instance(self.ec2, instance_name, 'cvoloshi', 'sg-000cca6b', 'p2.xlarge', 'ami-0796d6b51f01ffd0c', 'subnet-be91fdd7', startup_script))
        #         count += 1
        #     except:
        #         pass
        inst = self.ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
        instances = []
        for instance in inst: 
            if instance.id == 'i-0036b619ee40acf4d': 
                instances.append([instance])
                break

        print('Able to create %s out of (desired) %s instances' % (len(instances), self.args.number_of_instances))

        
        instances = [x[0] for x in instances]
        self.instances = instances
        self.sshs = {}
        self.tracker = {instance:{'start_time':0, 'current_task':None, 'in_queue':False, 'ready': False, 'attempt':0, 'completed':0} for instance in instances}
        # time.sleep(2*60)



    def print_stats(self):
        df = pd.DataFrame(self.tracker).T
        df['run_time'] = time.time() - df['start_time']
        print(df[['current_task', 'run_time', 'attempt', 'completed']][df['current_task'] >= 0])

    def run(self):
        try:
            self.loop = asyncio.get_event_loop()
            self.available_workers = asyncio.Queue(loop=self.loop)
            # self.session = aiobotocore.get_session(loop=self.loop)
            # self.s3 = self.session.create_client('s3', region_name='us-east-2',
            #                            aws_secret_access_key=self.args.secret,
            #                            aws_access_key_id=self.args.access)
            producer_coro = self.produce()
            consumer_coro = self.consume()
            self.starttime = time.time()
            self.loop.run_until_complete(asyncio.gather(producer_coro, consumer_coro))
            self.loop.close()
        finally:
            self.breakdown()
    
    def experiment_done(self, current_task):
        try:
            self.s3.Object('mc-pixel-ope-experiments', '/'.join(['exp_%05d' % current_task,'results.json']) ).load()
            return True
        except:
            return False
    
    def is_available(self, worker):
        current_task = self.tracker[worker]['current_task']
        if not self.tracker[worker]['in_queue']:
            
            if not self.tracker[worker]['ready']:
                statuses = self.client.describe_instance_status(InstanceIds=[worker.id])
                
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
                    if (time.time() - self.tracker[worker]['start_time']) > 60*7:
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

        experiments_already_ran = set(read_s3_bucket(self.s3, 'mc-pixel-ope-experiments'))
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

        
        user_name='ubuntu'
        pem_addr= os.path.join('secret.pem') # folder path to aws instance key
        aws_region='us-east-2' 


        if instance not in self.sshs:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            privkey = paramiko.RSAKey.from_private_key_file(pem_addr)
            failed = 0
            
            print(instance.public_dns_name)
            ssh.connect(instance.public_dns_name, username=user_name, pkey=privkey)
            self.sshs[instance] = [ssh]
        else:
            ssh = self.sshs[instance][0]
        #you can seperate two shell commands by && or ;
        cmd_to_run='cd fqe; pipenv run python run_mc.py {0} {1} {2} {3} {4}'.format(*self.hyperparams[exp_num].values())
        cmd_to_run = cmd_to_run + ' {0} {1} {2}'.format(exp_num, self.args.access, self.args.secret)
        stdin4, stdout4, stderr4 = ssh.exec_command(cmd_to_run,timeout=None, get_pty=False)
        self.sshs[instance] = [self.sshs[instance][0]] + [stdin4, stdout4, stderr4]
        output = ''
        for line in stdout4: output+=line
        print(output)
        output = ''
        for line in stderr4: output+=line
        print(output)

    def breakdown(self):
        # time.sleep(5)
        # for key, ssh in self.sshs.items():
        #     ssh[0].close()

        # for instance in self.instances:
        #     instance.terminate()
        pass

def distribute_experiments(args):
    '''
        Splits the task into N number of parallel data creation tasks. Uploads data files to S3.
    '''
    session = boto3.Session(aws_access_key_id=args.access, aws_secret_access_key=args.secret, region_name='us-east-2')
    s3 = session.resource('s3')
    ec2 = session.resource('ec2')
    client = session.client('ec2')

    
    hyperparams = create_experiment_metadata()
    with open('mc_experiment_hyperparam.json', 'w') as outfile:
        json.dump(hyperparams, outfile)

    runner = TaskRunner(client, ec2, s3, hyperparams, args)
    runner.run()

    
    
if __name__ == '__main__':
    print('start')
    parser = argparse.ArgumentParser(description='Distribute experiments across ec2 instances.')
    
    parser.add_argument('access', help='AWS Access Key')
    parser.add_argument('secret', help='AWS Secret Access Key')
    
    parser.add_argument('-N', dest='number_of_instances', default=1, required=False, type=int)
    

    args = parser.parse_args()

    distribute_experiments(args)
        
