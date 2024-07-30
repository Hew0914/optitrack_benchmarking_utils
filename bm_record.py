import cv2
import sys
import os
import re
import csv
import threading
from pynput import keyboard
from pathlib import Path
from optitrack_utils.NatNetClient import NatNetClient

# This program grabs frame-by-frame pose information on a single from an Optitrack client using the NatNet SDK
# While every Optitrack frame is recorded, the camera is read only every eigth frame.
# File is saved on shutdown
# For .traj files, time is estimated based on frame number, with frame zero occuring at 0.00s
# Frames are saved to folder
# To run the client, simply run on command line

FILE_OPTIONS = ['TRAJ', 'CSV', 'TXT']

# Variables

RECEIVING_CLIENT_ADDRESS = '127.0.0.1'
OPTITRACK_SERVER_ADDRESS = '192.168.2.109'
MULTICAST = True
OPENCV_CAMERA_SOURCE = 0
FILETYPE = 2

# The Optitrack client saves both the camera and 
class OptiTrackClient:
    def __init__(self):
        self.capture = cv2.VideoCapture(OPENCV_CAMERA_SOURCE)
        self.is_recording = False
        self.frame_number = None
        self.rigid_body_positions = []
        self.lock = threading.Lock()
        self.natnet = None
        self.filetype = FILE_OPTIONS[FILETYPE]
        self.recording_name = input("Enter the recording name: ")
        os.makedirs(self.recording_name, exist_ok=True)

    def setup(self):
        pass

    def start_recording(self):
        return_code = self.natnet.send_command('SetPlaybackStartFrame=0')
        if return_code == -1:
            print('Failed to start recording')
        else:
            self.is_recording = True
            print(f'Recording started.')

    def stop_recording(self):
        self.is_recording = False
        print(f'Recording stopped.')

    def take_photo(self):
        ret, frame = self.capture.read()
        if ret:
            photo_filename = f'photo_{self.frame_number}.png'
            cv2.imwrite(Path(f'{self.recording_name}/frames') / photo_filename, frame)
            print(f'Saved {photo_filename}')

    def record_position(self, rigid_body):
        if self.frame_number % 4 == 0:
            position_data = {
                'frame_number': self.frame_number,
                'data': rigid_body
            }
            self.rigid_body_positions.append(position_data)
            print(f'Recorded pose for frame {self.frame_number}: {rigid_body}')

    # Frame listener attached to NatNet client
    def on_frame(self, data_dict):
        with self.lock:
            # print('frame has lock')
            self.frame_number = data_dict['frame_number']
            if self.is_recording and self.frame_number % 4 == 0:
                self.take_photo()

    # Rigid body listener attached to NatNet client
    def on_rigid_body(self, id, position, rotation):
        with self.lock:
            # print('rigid body has lock')
            rigid_body_data = position + rotation
            if self.is_recording:
                self.record_position(rigid_body_data)

    def write_csv(self):
        with open(Path(self.recording_name) / 'optitrack_untimed.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for entry in self.rigid_body_positions:
                row = [entry['frame_number']] + list(entry['data'])
                writer.writerow(row)
            f.close()
    
    def write_traj(self):
        with open(Path(self.recording_name) / 'optitrack_untimed.traj', 'w', newline='') as f:
            f.write('#name optitrack\n')
            for entry in self.rigid_body_positions:
                row = [entry['frame_number']] + list(entry['data'])
                row_string = re.sub(r'[\[\]]', '', str(row))
                f.write(f'{row_string}\n')
            f.close()

    def write_txt(self):
        with open(Path(self.recording_name) / 'optitrack_untimed.txt', 'w', newline='') as f:
            for entry in self.rigid_body_positions:
                row = [entry['frame_number']] + list(entry['data'])
                line = ' '.join(map(str, row))
                f.write(f'{line}\n')
            f.close()

    def run(self):
        # Create and configure the NatNet client
        client = NatNetClient()
        self.natnet = client
        client.new_frame_listener = self.on_frame
        client.rigid_body_listener = self.on_rigid_body
        client.set_client_address(RECEIVING_CLIENT_ADDRESS)
        client.set_server_address(OPTITRACK_SERVER_ADDRESS)
        client.set_use_multicast(MULTICAST)

        # Make directories to save data and check camera function
        os.makedirs(f'{self.recording_name}/frames/', exist_ok=True)
        self.natnet.set_print_level(0)
        if not self.capture.isOpened():
            print("Could not open webcam.")
            sys.exit(1)
        
        # Start NatNet client
        is_running = client.run()
        if not is_running:
            print('Failed to start running')
            sys.exit(1)

    # Closes the NatNet client and saves data based on file option
    def shutdown(self):
        with self.lock:
            if self.filetype == FILE_OPTIONS[0]:
                self.write_csv()
            elif self.filetype == FILE_OPTIONS[1]:
                self.write_traj()
            elif self.filetype == FILE_OPTIONS[2]:
                self.write_txt()
            self.natnet.shutdown()
            print("shut off")

    def on_press(self, key):
        try:
            if key.char == 's':  # Start recording on 's' key press
                self.start_recording()
            elif key.char == 'e':  # Stop recording on 'e' key press
                self.stop_recording()
            elif key.char == 'q': # Quit client and save results on 'q' key press
                self.shutdown()
        except AttributeError:
            pass

    def listen_for_keypress(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

if __name__ == '__main__':
    client = OptiTrackClient()
    threading.Thread(target=client.listen_for_keypress).start()
    client.run()
