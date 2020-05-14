import boto3
import os
import json
from requests import get
import sys
import time
import argparse
import math



# configuring aws group, role and user settings and the port configurations for each instance
def cloud_setup():
    iam = boto3.client('iam')
    client = boto3.client('ec2')
    ec2 = boto3.resource('ec2')
    ip = get('https://api.ipify.org').text
    print('Your IP Address: ' + ip)

    # creating a group for user/roles to be assigned to
    print("Starting setup...")
    print("Creating group...")
    response = iam.create_group(
        Path="/",
        GroupName="admin"
    )
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "1",
                "Effect": "Allow",
                "Principal": {
                    "Service": "ec2.amazonaws.com"
                },
                "Action": [
                    "sts:AssumeRole"
                ]
            }
        ]
    }
    iam.attach_group_policy(
        GroupName="admin",
        PolicyArn="arn:aws:iam::aws:policy/AdministratorAccess"
    )
    print("Group created.")

    # creating a role so that a user can assume
    # role has full administrator
    print("Creating identity access management role...")
    response = iam.create_role(
        RoleName='demo',
        AssumeRolePolicyDocument=json.dumps(trust_policy)
    )
    response = iam.attach_role_policy(
        RoleName='demo',
        PolicyArn='arn:aws:iam::aws:policy/AdministratorAccess'
    )
    response = iam.attach_role_policy(
        RoleName='demo',
        PolicyArn='arn:aws:iam::aws:policy/AmazonSSMFullAccess'
    )
    print("Role created.")

    # creating a profile for instances
    # there is a race condition for when an instance profile is created have to wait a set period
    # of time before you can create an instance with it
    print("Creating instance profile...")
    response = iam.create_instance_profile(
        InstanceProfileName="demoprofile"
    )
    instance_profile_arn = response['InstanceProfile']['Arn']
    response = iam.add_role_to_instance_profile(
        InstanceProfileName="demoprofile",
        RoleName="demo"
    )
    print("(will take approximately 20 seconds)")
    time.sleep(20)
    print("Instance profile created.")

    # creating security groups for the instances opens ports
    print("Creating security group for instance inbound traffic...")
    response = client.create_security_group(
        Description='demo secutrity group for the coursework should allow all outside connections',
        GroupName='demo_group'
    )
    response = client.authorize_security_group_ingress(
        GroupName='demo_group',
        IpPermissions=[
            {
                'FromPort': 22,
                'ToPort': 22,
                'IpProtocol': 'TCP',
                'IpRanges': [
                    {
                        'CidrIp': ip + '/32',
                        'Description': 'Allow inbound SSH access to Linux instances from IPv4 IP addresses in your network (over the Internet gateway)'
                    }
                ]
            },
            {
                'FromPort': 22,
                'ToPort': 22,
                'IpProtocol': 'TCP',
                'IpRanges': [
                    {
                        'CidrIp': '0.0.0.0/0',
                        'Description': 'Allow inbound SSH access to Linux instances from IPv4 IP addresses in your network (over the Internet gateway)'
                    }
                ]
            }
        ]
    )
    print("Security group created.")

    # creating a key to ssh into instances then saves it to the current directory
    key_name = 'aw16997-keypair'
    print("Creating Key Pair " + key_name + "...")
    outfile = open('aw16997-keypair.pem', 'w')
    key_pair = ec2.create_key_pair(KeyName=key_name)
    key_pair_out = str(key_pair.key_material)
    outfile.write(key_pair_out)
    outfile.close()
    os.system("chmod 400 aw16997-keypair.pem")
    print(key_name + " created and saved to local directory.")

    print("Setup complete.")

# starts up aws instances
def start_instances(filename, no_instances=1):
    iam = boto3.client('iam')
    client = boto3.client('ec2')
    ec2 = boto3.resource('ec2')
    key_name = 'aw16997-keypair'
    print("Starting instances...")

    # script that is run during the instance start up have to install python and start up an SSM agent to send
    # remote commands
    user_data_script = """#!/bin/bash
    cd /tmp
    sudo yum update
    touch i_made_it.txt
    sudo yum install -y https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/linux_386/amazon-ssm-agent.rpm
    cd ~
    sudo yum -y install python3
    sudo pip3 -q install twint
    sudo pip3 -q install pandas
    sudo pip3 -q install boto3
    sudo systemctl start amazon-ssm-agent
  
    """
    # creates instances
    instances = ec2.create_instances(
        ImageId='ami-00e8b55a2e841be44',
        MinCount=1,
        MaxCount=no_instances,
        InstanceType='t2.micro',
        KeyName=key_name,
        Monitoring={
            'Enabled': True
        },
        IamInstanceProfile={
            # "Arn": instance_profile_arn,
            "Name": "demoprofile"

        },
        SecurityGroups=[
            'demo_group',
        ],
        UserData=user_data_script

    )

    # waits until each instance is up and copies proof of work to each instance
    count = 0
    for instance in instances:
        print("Starting Instance-" + str(count) + "...")
        print("(Waiting until the instance is up and running, this may take a few moments)")
        instance.wait_until_running()
        instance.load()
        ip_address = instance.public_ip_address.replace('.', '-')
        print("Instance-" + str(count) +
              " IP Address: " + instance.public_ip_address)
        time.sleep(15)
        os.system('scp -oUserKnownHostsFile=/dev/null  -o LogLevel=ERROR -o "StrictHostKeyChecking no" -i %s %s ec2-user@ec2-%s.eu-west-2.compute.amazonaws.com:%s' %
                  ('aw16997-keypair.pem', filename, ip_address, filename))
        count += 1

    # checking all instances are ok
    print("Checking status of all instances...")
    waiter = client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=get_instance_ids(instances))
    print("Check complete.")

    return instances

# gets the ids each instance in a list of instances
def get_instance_ids(instances):
    ids = []
    for instance in instances:
        ids.append(instance.instance_id)
    return ids

# terminates each instance within a list of instances
def terminate_instances(instances):
    for instance in instances:
        instance.terminate()
    print("Instances have been terminated.")

# sends a unix command to an instance
def send_command_to_instance(instance, instance_no, commands):
    client = boto3.client('ssm')
    instance_id = instance.instance_id
    response = client.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Comment="python test",
        Parameters={
            'commands': commands,
            'executionTimeout': ["172800"]
        },
    )
    print("Commands sent to Instance-" + str(instance_no) + ".")

# sends all generated commands to instances
def send_all_commands(instances, commands):
    count = 0
    for i in instances:
        send_command_to_instance(i, count, [commands[count]])
        count += 1

# gets csv files from an instance
def get_files_from_instance(filename, instance):
    
    instance.load()
    ip_address = instance.public_ip_address.replace('.', '-')
    os.system('scp -oUserKnownHostsFile=/dev/null  -o LogLevel=ERROR -o "StrictHostKeyChecking no" -i %s ec2-user@ec2-%s.eu-west-2.compute.amazonaws.com:/usr/bin/%s .' %
              ('aw16997-keypair.pem', ip_address, filename))

# gets the output from an instance
def get_command_outputs(instances):
    # returns as soon as the first vm recieves an output
    client = boto3.client('ssm')
    completion = 0
    wait = True
    no_instances = len(instances)
    

    # waits until an instance produces an putput
    while wait:
        
       
        if completion == no_instances: wait = False
        completion_string = f"{completion}/{no_instances} instances are finished."
        retries = 0
        max_retries = 11
        print(completion_string)

        # checking each instacne fora response
        for i in instances:
            
            # exponential back off for api call to list command invocations
            wait_time = pow(2, retries) * 0.1
            time.sleep(wait_time)
            
            response = client.list_command_invocations(
                InstanceId=i.instance_id,
                Details=True
            )
            if (retries < max_retries): retries += 1
            
            # handles the response produced from an instance
            cinvs = response['CommandInvocations']
            if (cinvs and completion < no_instances):
                status = response['CommandInvocations'][0]['Status']
                
                if status == "Success":
                    
                    output = response['CommandInvocations'][0]['CommandPlugins'][0]['Output']
                    print(output.strip())
                    get_files_from_instance("*.csv", i)
                    i.terminate()
                    instances.remove(i)
                    completion += 1

                elif status == "Failed":

                    output = response['CommandInvocations'][0]['CommandPlugins'][0]['Output']
                    print(output.strip())
                    print("Instance failed to run command.")
                    get_files_from_instance("*.csv", i)
                    i.terminate()
                    instances.remove(i)
                    completion += 1

                    
                elif status == "InProgress":
                    print(f".")

                elif status == 'TimedOut':

                    wait = False
                    print("Instance Timed Out when running commands.")
                    get_files_from_instance("*.csv", i)
                    i.terminate()
                    instances.remove(i)
                    completion += 1
                    # print("Stopping all Instances change the timeout value for running the command.")
                
        
        retries = 6

    return

# generates commands depending on the instance number
def generate_commands(filename, number_of_vms):

    commands = []
   
    # produces the unix command as a string to send to each intsance
    for i in range(number_of_vms):
        command = f"python3 /home/ec2-user/{filename} {i}"
        commands.append(command)
    
    return commands

def main():
    
    filename = "twitter.py"
    number_of_vms = 31

    # run the cloud setup if you cant find a key in the current directory
    if not os.path.exists('aw16997-keypair.pem'):
        cloud_setup()

    instances = start_instances(filename, number_of_vms)
    commands = generate_commands(filename, number_of_vms)
    send_all_commands(instances, commands)
    outputs  = get_command_outputs(instances)
    terminate_instances(instances)

if __name__ == "__main__":
    main()
