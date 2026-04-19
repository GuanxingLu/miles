# python3.7
"""Launches a job at AI Studio.

This script defines the pipeline of launching the job on AI Studio. Note that
all commands in this file are executed on AI Studio instead of local machine.
"""

import os
import time
import sys
import socket
import psutil
# os.environ["IDE_COMMON_OSS_BUCKET"] = "antsys-gpu-beijing-code"
# os.environ["ENV_ENCRYPTED_SECRET"] = "w1ODdlZjFlZTRlYzU4YThi"
from aistudio_common.utils.global_param_utils import GlobalParamUtils


def find_free_port():
    """Finds a free port in the machine.

    Returns:
        An integer, representing the port that is free to use.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 0))
    sockname = sock.getsockname()
    sock.close()
    return int(sockname[1])


def get_ip(network_card_name='eth0'):
    """Gets the IP address of the given network card.

    Args:
        network_card_name: Name of the network card. (default: `eth0`)

    Returns:
        A string, such as `127.0.0.1`, representing the IP address.
    """
    addresses = {}
    # Find all addresses that can be used as hosts.
    for card_name, card_info in psutil.net_if_addrs().items():
        # `card_info` is a list of `snicaddr`.
        for snicaddr in card_info:
            # Each `snicaddr` is organized as
            #   (family, address, netmask, broadcase, ptp).
            if snicaddr[0] == 2:
                # 2 means `family == AddressFamily.AF_INET`, which can be used
                # as host.
                addresses[card_name] = snicaddr[1]
    return addresses[network_card_name]


def get_master_ip_port(node_rank, cache_path, network_card_name='eth0'):
    """Gets IP address and port of the master node.

    The master node will save its IP address and port onto the disk, which can
    be accessed by all other nodes. The other nodes will wait until the
    information is successfully saved by the master node, and then load the
    information.

    Args:
        node_rank: Node rank, which is used to recognize the master, i.e., 0.
        cache_path: Path to cache the information of the master node.
        network_card_name: Name of the network card. (default: `eth0`)

    Returns:
        A two-element tuple, indicating the IP address (string) and the port
            (integer) of the master node, respectively.
    """
    if node_rank == 0:  # master node
        # Get IP address and port on the master node.
        master_ip = get_ip(network_card_name)
        master_port = find_free_port()
        # Save the master information.
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'w') as f:
            f.write(f'{master_ip}:{master_port}')
    else:  # other nodes
        # Wait until the master information is successfully saved.
        while not os.path.exists(cache_path):
            time.sleep(10.0)
        # Load the information of the master node.
        with open(cache_path, 'r') as f:
            info = f.read()
        master_ip, master_port = info.strip().split(':')

    return master_ip, int(master_port)


def run_cmd(cmd, log_cmd=False):
    """Prints and executes a system command.

    Args:
        cmd: Command to be executed.
        log_cmd: Whether to log the command. This flag should be carefully used
            since it may cause information leak, such as passwords.
            (default: False)
    """
    if log_cmd:
        print(f'\nRUN COMMAND: {cmd}\n', flush=True)
    os.system(cmd)

def run_cmd_last(cmd, log_cmd=False):
    """Prints and executes a system command.

    Args:
        cmd: Command to be executed.
        log_cmd: Whether to log the command. This flag should be carefully used
            since it may cause information leak, such as passwords.
            (default: False)
    """
    if log_cmd:
        print(f'\nRUN COMMAND: {cmd}\n', flush=True)
    code_exit = os.system(cmd)
    if code_exit == 0:
        sys.exit(0)
    else:
        sys.exit(1)

def main():
    """Main function."""
    args = GlobalParamUtils.get_job_param()

    # List environment.
    run_cmd('ifconfig', log_cmd=True)
    run_cmd('nvidia-smi', log_cmd=True)

    oss_bucket = args['oss_bucket']
    oss_domain = args['oss_domain']
    oss_key_id = args['oss_key_id']
    oss_key_secret = args['oss_key_secret']

    # Configure OSS storage.
    run_cmd('wget https://gosspublic.alicdn.com/ossutil/1.7.13/ossutil64 && '
            'sudo mv ossutil64 /usr/local/bin/', log_cmd=True)
    run_cmd('chmod +x /usr/local/bin/ossutil64', log_cmd=True)
    run_cmd(f'ossutil64 config '
            f'-e {oss_domain} '
            f'-i {oss_key_id} '
            f'-k {oss_key_secret}')

    os.environ["OSS_BUCKET"] = oss_bucket
    run_cmd(f'ossutil64 cp oss://{oss_bucket}/vilab/oss_manager-1.0.0-py3-none-any.whl ./', log_cmd=True)
    # The miles image ships pyparsing 3.1.1 via dpkg. pip cannot uninstall
    # dpkg-managed packages ("no RECORD file was found"), so installing
    # oss_manager — which transitively pulls wfbuilder → pyparsing — blows up.
    # Drop the dpkg copy first so pip can manage pyparsing cleanly.
    run_cmd('apt-get remove -y python3-pyparsing || true', log_cmd=True)
    run_cmd('pip install --ignore-installed pyparsing oss_manager-1.0.0-py3-none-any.whl', log_cmd=True)

    # Mount NAS storage.
    if args["nas_stores"]:
        # pylint: disable=line-too-long
        NAS_MNT_COMMAND = (
            'mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport'
        )
        # # pylint: enable=line-too-long
        for nas in args["nas_stores"].values():
            run_cmd(f'mkdir {nas.nas_mnt_path} && '
                    f'{NAS_MNT_COMMAND} {nas.nas_domain} {nas.nas_mnt_path}', log_cmd=True)
        run_cmd('ln -s /input /personal', log_cmd=True)

        
    # Register docker image ID as an environment variable.
    os.environ['DOCKER_IMAGE_ID'] = args['image']

    # Set `g++` as the default compiler to avoid PyTorch warning.
    os.environ['CXX'] = 'g++'

    # Launch job.
    command = args['command']
    runtime = args['runtime']


    CODE_PATH = f'/workspace/bin/{args["repo_url"].split("/")[-1]}'

    if runtime=='easydl':
        '''
        EasyDL Job 会给每个 Pod 配置如下环境变量：
        ● NODE_NUM: 用户配置的 worker 节点(Pod)的数量。
        ● NODE_RANK: worker Pod 在节点集合中的节点 rank。
        ● NODE_ID: worker Pod 的序号。
        注：EasyDL Job 并没有配置 PyTorch 组网训练需要的 MASTER_ADDR, MASTER_PORT, WORLD_SIZE 和 RANK, 因为 dlrover-run 采用的是 TorchElastic 动态组网的方案，会在启动时自动配置到各个进程的环境变量中。
        '''
        run_cmd_last(f'cd {CODE_PATH} && {command}', log_cmd=True)
    else:
        node_size = os.getenv('WORLD_SIZE')  # `WORLD_SIZE` handled by AI Studio.
        if node_size == 1:  # single node
            run_cmd_last(f'cd {CODE_PATH} && {command}', log_cmd=True)
        else:  # multi-node
            job_id_cache_dir = args['job_id_cache_dir']
            job_id = args['job_id']
            node_rank = int(os.getenv('RANK'))  # `RANK` handled by AI Studio.
            cache_path = os.path.join(f'{job_id_cache_dir}',
                                      f'job_{job_id}_master_ip_port.txt')
            network_card_name = args['network_card_name']
            master_ip, master_port = get_master_ip_port(
                node_rank, cache_path, network_card_name)
            env_vars = (f'export MASTER_IP={master_ip} &&'
                        f'export MASTER_PORT={master_port} &&'
                        f'export NODE_SIZE={node_size} &&'
                        f'export NODE_RANK={node_rank} &&')
            run_cmd_last(f'cd {CODE_PATH} && {env_vars} {command}', log_cmd=True)


if __name__ == '__main__':
    main()