# Copyright (c) 2023 Huazhong University of Science and Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Muyuan Shen <muyuan_shen@hust.edu.cn>


import ns3ai_apb_py_vec as py_binding
from ns3ai_utils import Experiment
import sys
import traceback

from tensorforce import Agent
import numpy as np

APB_SIZE = 3

exp = Experiment("cloudnet  --cwd=scratch/cloudnet/cfg -- --Prefix=example", "../../../../../", py_binding,
                 handleFinish=True, useVector=True, vectorSize=APB_SIZE)
msgInterface = exp.run(show_output=True)

# Instantiate a Tensorforce agent
agent = Agent.create(
            agent='tensorforce',
            states=dict(type='float', shape=(APB_SIZE,)),
            actions=dict(type='int', shape=(APB_SIZE,), num_values=3),
            #environment=environment  # alternatively: states, actions, (max_episode_timesteps)
            memory=10000,
            update=dict(unit='timesteps', batch_size=64),
            optimizer=dict(type='adam', learning_rate=3e-4),
            policy=dict(network='auto'),
            objective='policy_gradient',
            reward_estimation=dict(horizon=20)
        )


try:
    while True:
        # receive from C++ side
        msgInterface.PyRecvBegin()
        if msgInterface.PyGetFinished():
            break

        # send to C++ side
        msgInterface.PySendBegin()

        # the agent expects the state as numpy array
        status = np.empty ([APB_SIZE])
        # we read the reward
        reward = msgInterface.GetCpp2PyVector()[0].b

        for i in range(len(msgInterface.GetCpp2PyVector())):
            status [i] = msgInterface.GetCpp2PyVector()[i].a
        print ("[Agent, status (get]:", end = ' ')
        print (status)
        print ("[Agent, reward (get)]:", end = ' ')
        print (reward)
        #for i in range(len(msgInterface.GetCpp2PyVector())):
            # calculate the sums
        #    msgInterface.GetPy2CppVector()[i].c = msgInterface.GetCpp2PyVector()[i].a # + msgInterface.GetCpp2PyVector()[i].b

        # AI algorithms here and put the data back to the action
        actions = agent.act(states=status, independent=False)
        agent.observe(terminal=False, reward=reward)
        print ("[Agent, actions (set)]:", end = ' ')
        print (actions)

        for i in range(len(msgInterface.GetCpp2PyVector())):
            # calculate the sums
            msgInterface.GetPy2CppVector()[i].c = actions [i] # + msgInterface.GetCpp2PyVector()[i].b

        msgInterface.PyRecvEnd()
        msgInterface.PySendEnd()

except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Exception occurred: {}".format(e))
    print("Traceback:")
    traceback.print_tb(exc_traceback)
    exit(1)

else:
    pass

finally:
    print("Finally exiting...")
    del exp
