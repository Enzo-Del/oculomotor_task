import pyfirmata
import time
import serial
def init_reward(com_port):
    board = pyfirmata.Arduino(com_port)
    print("Communication Successfully started")
    lick_GPIO = board.digital[2]
    reward_GPIO = board.digital[6]
    reward_GPIO.write(0)
    it = pyfirmata.util.Iterator(board)
    it.start()
    lick_GPIO.mode = pyfirmata.INPUT
    lick_state = lick_GPIO.read()

    return board, lick_GPIO, reward_GPIO, it

def give_reward(reward_GPIO, time_ON):
    reward_GPIO.write(1)
    time.sleep(time_ON)
    reward_GPIO.write(0)

def check_lick(lick_GPIO):

    if lick_GPIO.read() == 1:
        lick =1
        time.sleep(0.05)

    else:
        lick = 0

    return lick


#main
# 5 ms = bonne goutte

#board, lick_GPIO, reward_GPIO, it = init_reward('COM4')

#while True :
    #lick = check_lick(lick_GPIO)

    #if lick == 1:
        #print(lick)



