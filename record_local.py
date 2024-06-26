import cv2
import sys
import time
from optitrack_utils import DataDescriptions
from optitrack_utils import MoCapData
from optitrack_utils.NatNetClient import NatNetClient

# This is a callback function that gets connected to the NatNet client
# and called once per mocap frame.
def receive_new_frame(data_dict):
    order_list=[ "frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
                "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged" ]
    dump_args = False
    if dump_args == True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "="
            if key in data_dict :
                out_string += data_dict[key] + " "
            out_string+="/"
        print(out_string)

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame( new_id, position, rotation ):
    print( "Received frame for rigid body", new_id," ",position," ",rotation )

def print_configuration(natnet_client):
    natnet_client.refresh_configuration()
    print("Connection Configuration:")
    print("  Client:          %s"% natnet_client.local_ip_address)
    print("  Server:          %s"% natnet_client.server_ip_address)
    print("  Command Port:    %d"% natnet_client.command_port)
    print("  Data Port:       %d"% natnet_client.data_port)

def initial_message():
    msg = "Commands for Local Recorder:"
    msg += "Begin Recording: b\n"
    msg += "End Recording: e\n"
    msg += "Change verbosity: j\n"
    msg += "Stop client: q\n"
    msg += "\nVideo saved automatically to script directory on recording end.\n"
    print(msg)

def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len=len(arg_list)
    if arg_list_len>1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len>2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len>3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict

# Recording Process; run script on terminal 
if __name__ == "__main__":

    # cam = cv2.VideoCapture(0)
    # if not cam.isOpened():
    #     print("ERROR: Could not open webcam. Exiting")
    #     sys.exit(1)
    #CHANGE CLIENT ADDRESS TO OPTITRACK COMPUTER
    optionsDict = {}
    optionsDict["clientAddress"] = "127.0.0.1"
    optionsDict["serverAddress"] = "192.168.2.109:1001"
    optionsDict["use_multicast"] = True

    # This will create a new NatNet client
    optionsDict = my_parse_args(sys.argv, optionsDict)

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.new_frame_listener = receive_new_frame
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client. Exiting.")
        sys.exit(1)

    is_looping = True
    time.sleep(1)
    if streaming_client.connected() is False:
        print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
        try:
            sys.exit(2)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    print_configuration(streaming_client)
    print("\n")
    initial_message()


    while is_looping:
        inchars = input('Enter command or (\'h\' for list of commands)\n')
        if len(inchars)>0:
            c1 = inchars[0].lower()
            if c1 == 's':
                request_data_descriptions(streaming_client)
                time.sleep(1)
            elif c1 == 'e':
                sz_command="TimelineStop"
                return_code = streaming_client.send_command(sz_command)
                time.sleep(1)
                print("Command: %s - return_code: %d"% (sz_command, return_code) )
            elif c1 == 'b':
                sz_command="TimelinePlay"
                return_code = streaming_client.send_command(sz_command)
                print("Command: %s - return_code: %d"% (sz_command, return_code) )

            elif c1 == 'j':
                if streaming_client.get_print_level == 1:
                    streaming_client.set_print_level(0)
                else:
                    streaming_client.set_print_level(1)
                print("Changed verbosity")
            elif c1 == 'q':
                is_looping = False
                streaming_client.shutdown()
                break
            else:
                print("Error: Command %s not recognized"%c1)
            print("Ready...\n")
    print("exiting")
