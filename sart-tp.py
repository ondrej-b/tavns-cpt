'''
Sustained Attention to Response Task with Thought Probes (SART-TP)
Version: 2.0
Updated: 2-27-2024
By: Nicholas C. Dunn
-------------------------------------------------------------------
# Description

This task was developed based on the Cristoff et al. (2021) study, which used a SART with Thought Probes to assess mindwandering.
We programmed this task for use in an fMRI machine. As such, frame-based timing was used to ensure millisecond accuracy for
the presentation of stimuli, which is necessary for fMRI studies.

*NOTE: This functionality (i.e., frame-based timing, non-slip timing) is difficult to achieve in PsychoPy builder. If all of your trials
are the same length, then it is possible to do, and you can find guides online for how to achieve this. However, our task needed
target and non-target stimuli presentations that were different total durations than our thought probe trials, so coding this task
was the only option (as of this version of PsychoPy). In theory, you could also achieve this via while loops and clock-based timing.
I just learned frame-based timing first.

The task presents a random number between 0 and 9 for 500ms followed by a 1800ms inter-stimulus interval (ISI) for target and non-target
trials. If the stimulus is a non-target trial (any number but 3), the participant is instructed to press the <left> key. If the stimulus is
3, the participant is instructed to without a response. Reaction time (ms) and whether they correctly responded are saved to an excel output.

Occasionally, a thought probe is presented. Probe 1 asks whether the participant's mind was on-task or off-task at the time of presentation. 
It is presented for 500ms, with an ISI of 0ms. If on-task, the participant responds by pressing the <left> key. If off-task, the participant 
responds by pressing the <right> key. The response to probe 1 is saved to the excel output. For probe 2, the participant is presented with
a conditional stimulus, depending on their response to probe 1. If on-task in probe 1, they are instructed to complete an attention check,
wherein they are told either to press the <left> or <right> key. If off-task or did not respond to probe 1, they are asked whether
they were externally distracted (<left>) or daydreaming/mindwandering (<right>). This response is also recorded in the excel output.

There are 5 blocks total, with 1725 total trials. Each block has 297 non-target trials, 16 target trials, and 16 thought probe pairs (probe 1
and probe 2), totaling 345 trials. Targets and thought probe pairs were pseudocounterbalanced to only occur with a random distance between
5-15 non-target trials away from each other (e.g., you receive a target trial --> you will receive another target OR probe between
5 and 15 non-target trials after finishing the target trial). Each block takes about 17 minutes.  
'''

import os
import sys
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, monitors
from psychopy.hardware import keyboard
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np
import pandas as pd


# Constants
PSYCHOPY_VERSION = '2021.2.2'
EXP_NAME = 'SART-TP'
FRAME_TOLERANCE = 0.0001
REFRESH_RATE = 1.0 / 60.0 # Edit if the refresh rate of your monitor is a different Hz.
NUM_ISI = 1.8 # Blank screen duration following a non-target/target trial
NUM_DURATION = 0.5 # Stimulus duration for a non-target/target trial
PROBE1_ISI = 3 # Blank screen duration following a probe 1 trial
PROBE1_DURATION = 8 # Stimulus duration for a probe 1 trial
PROBE2_ISI = 3 # Blank screen duration following a probe 2 trial
PROBE2_DURATION = 8 # Stimulus duration for a probe 2 trial
PRACTICE_INSTRUCTIONS = [
    'Vítejte v úloze SART-TP.',
    'V této úloze uvidíte čísla od 0 do 9. Vaším úkolem je stisknout ← levou šipku pokaždé, když se na obrazovce objeví jakékoli číslo KROMĚ čísla 3.\n Pokud se objeví číslo 3, ← levou šipku nestiskněte.',
    'Občas se může objevit otázka, zda byla vaše pozornost zaměřena na úkol. \n Pokud ano, stiskněte ← levou šipku. \n\n Pokud vaše pozornost nebyla zaměřena na úkol, stiskněte → pravou šipku.',
    'Pokud bude vaše pozornost zaměřena na úkol, budete vyzván/a ke stisknutí buď ← levé šipky, nebo → pravé šipky.',
    'Pokud vaše pozornost nebyla zaměřena na úkol, budete dotázán/a, zda byla vaše pozornost rozptýlena vnějšími podněty ← levá šipka, nebo zda jste se zasnil/a → pravá šipka.'
]

INSTRUCTIONS = [
    'Vítejte v SART-TP',
    'Úloha začne za okamžik. Prosím vyčkejte.'
]


# Setup monitorů
MON_NAME = 'taVNS_lab'

if MON_NAME not in monitors.getAllMonitors():
    sys.exit(f"Monitor '{MON_NAME}' not calibrated on this machine — run setup_monitor.py first.")

mon = monitors.Monitor(MON_NAME)
if mon.getWidth() is None or mon.getDistance() is None:
    sys.exit(f"Monitor '{MON_NAME}' is missing width or distance.")

win = visual.Window(monitor=mon, units='deg', fullscr=True)

# Initial Setup
this_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_dir)
exp_info = {'participující': '', 'sezení': '1', 'practice': ('No', 'Yes')}
dlg = gui.DlgFromDict(dictionary = exp_info, sortKeys=False, title=EXP_NAME)
if not dlg.OK:
    core.quit()
exp_info['date'] = data.getDateStr(format="%Y-%m-%d-%H%M")
exp_info['expName'] = EXP_NAME
exp_info['psychopyVersion'] = PSYCHOPY_VERSION
if exp_info['practice'] == "No":
    filename = this_dir + os.sep + f'data/{exp_info["participující"]}_{EXP_NAME}_{exp_info["date"]}_{exp_info["sezení"]}'
elif exp_info['practice'] == "Yes":
    filename = this_dir + os.sep + f'data/{exp_info["participující"]}_{EXP_NAME}_{exp_info["date"]}_PRACTICE'

# Experiment Handler - handles saving data during the task.
thisExp = data.ExperimentHandler(
    name=EXP_NAME, version='',
    extraInfo=exp_info, runtimeInfo=None,
    originPath=os.path.abspath(__file__),
    savePickle=True, saveWideText=True,
    dataFileName=filename
)
log_file = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)

# Window Setup
monitor = monitors.Monitor('monitor')
window = visual.Window(size=(1920, 1080), fullscr=True)
exp_info['frameRate'] = window.getActualFrameRate()
frame_dur = 1.0 / round(exp_info['frameRate']) if exp_info['frameRate'] else REFRESH_RATE

# Stimuli
kb = keyboard.Keyboard()
instr_stim = visual.TextStim(window)
starting_stim = visual.TextStim(window, 'Začínáme za', font='Open Sans', pos=(0, .5))
break_stim = visual.TextStim(window, 'PAUZA', font='Open Sans', pos=(0, .5))
complete_stim = visual.TextStim(window, 'Dokončil/a jste úlohu.\n\nDěkujeme!', font='Open Sans', pos=(0, 0))
stim = visual.TextStim(window)
probe1_resp1 = visual.TextStim(window, 'Na úlohu', font='Open Sans', pos=(-.5,-.5))
probe1_resp2 = visual.TextStim(window, 'Na něco jiného', font='Open Sans', pos=(.5,-.5))
probe2_resp1 = visual.TextStim(window, 'Levá šipka', font='Open Sans', pos=(-.5,-.5))
probe2_resp2 = visual.TextStim(window, 'Pravá šipka', font='Open Sans', pos=(.5,-.5))
probe2_resp3 = visual.TextStim(window, 'Byla rozptýlena vnějšími podněty', font='Open Sans', pos=(-.5,-.5))
probe2_resp4 = visual.TextStim(window, 'Byla zasněná', font='Open Sans', pos=(.5,-.5))
vertical_line = visual.TextStim(window, '|', font='Open Sans', pos=(0, -.5))

# Frames
num_frames = int(NUM_DURATION / REFRESH_RATE)
num_isi_frames = int(NUM_ISI / REFRESH_RATE)
total_num_frames = int(num_frames + num_isi_frames)
probe1_frames = int(PROBE1_DURATION / REFRESH_RATE)
probe1_isi_frames = int(PROBE1_ISI / REFRESH_RATE)
total_probe1_frames = int(probe1_frames + probe1_isi_frames)
probe2_frames = int(PROBE2_DURATION / REFRESH_RATE)
probe2_isi_frames = int(PROBE2_ISI / REFRESH_RATE)
total_probe2_frames = int(probe2_frames + probe2_isi_frames)

# Counts
block_count = 1

# Global Clock
global_clock = core.Clock()
timestamp_clock = core.Clock()

def display_instructions(instructions):
    '''
    Displays a list of instruction slides, advancing to the next slide when the
    participant presses the <right> key. Pressing <escape> quits the task.

    Parameters:
    instructions (list): a list of strings, where each string is an instruction slide.
    '''
    core.wait(5) # Wait for program to be set to fullscreen before starting instructions
    for instr in instructions:
        instr_stim.setText(instr)
        instr_stim.draw()
        window.flip()
        keys = event.waitKeys(keyList=['right', 'escape'])
        if 'escape' in keys:
            window.close()
            core.quit()

def display_break(start_number, block_count):
    '''
    Displays a countdown break from the specified start number down to 1.
    
    Parameters:
    start_number (int): the number from which the countdown starts.
    block_count (int): the number of the current block.
    '''
    for sec in range(start_number, 0, -1):
        instr_stim.setText(str(sec))
        break_timer = core.CountdownTimer(1)
        while break_timer.getTime() > 0:
            if block_count == 1:
                starting_stim.draw()
            elif block_count > 1:
                break_stim.draw()
            instr_stim.draw()
            window.flip()
    
def display_blank():
    '''
    Displays a blank screen for 1 second.
    
    Parameters:
    None
    '''
    blank_timer = core.CountdownTimer(5)
    while blank_timer.getTime() > 0:
        window.flip()
        
def display_complete():
    complete_timer = core.CountdownTimer(10)
    while complete_timer.getTime() > 0:
        complete_stim.draw()
        window.flip()
            
def initialize_trial_handler(block):
    '''
    Initializes the PsychoPy trial handler for the specified block condition file.
    
    Parameters:
    block (str): the string for the block condition file in the list of blocks.
    '''
    return data.TrialHandler(nReps=1, method='sequential', trialList=data.importConditions(block))
    
def add_trial_data(trial, stim_onset, end_time):
    '''
    Handles adding trial data to excel output file.
    
    Parameters:
    trial (dict): a dictionary containing trial information.
    '''
    thisExp.addData('block', block_count)
    thisExp.addData('trial', trial_count)
    thisExp.addData('trial_type', trial['trialType'])
    thisExp.addData('stimulus', trial['stimulus'])
    thisExp.addData('stim_onset', stim_onset)
    thisExp.addData('trial_duration', end_time)
            
def run_number_trial(trial, trial_clock, total_num_frames, num_frames):
    '''
    Handles the logic for a number rial, including displaying stimuli and recording responses.
    
    Parameters:
    trial (dict): a dictionary containing trial information.
    trial_clock (function): a Clock function from Psychopy.
    total_num_frames (int): the total number of frames for the trial.
    num_frames (int): the number of frames the stimulus is displayed.
    '''
    trial_clock.reset()
    correct=0
    sub_resp = None
    kb.clock.reset()
    for frame_n in range(total_num_frames):
        if 0 <= frame_n < num_frames:
            stim.setText(trial['stimulus'])
            stim.draw()
            window.flip()
            if frame_n == 0:
                stim_onset = global_clock.getTime() * 1000
        elif num_frames <= frame_n < total_num_frames:
            window.flip()
    keys = kb.getKeys(['left'])
    rt = None
    timestamp = None
    for key in keys:
        if key.name == 'left':
            rt = key.rt*1000
            sub_resp = key.name
            correct = 1 if key.name == trial['corrAns'] else 0
            timestamp = (float(rt) + float(stim_onset))
            break
        
    #timestamp = float(rt) + float(stim_onset)     
    if trial['trialType'] == "Target" and not keys:
        correct = 1
    
    end_time = trial_clock.getTime() * 1000            
    add_trial_data(trial, stim_onset, end_time)
    thisExp.addData('rt', rt)
    thisExp.addData('timestamp', timestamp)
    thisExp.addData('response', sub_resp)
    thisExp.addData('correct', correct)
    
def run_probe1_trial(trial, trial_clock, total_probe1_frames, probe1_frames):
    '''
    Handles the logic for a probe1 trial, including displaying stimuli and recording responses.
    
    Parameters:
    trial (dict): a dictionary containing trial information.
    trial_clock (function): a Clock function from Psychopy.
    total_probe1_frames (int): the total number of frames for the probe1 display and response.
    probe1_frames (int): number of frames that the probe1 stimulus is displayed.
    '''
    trial_clock.reset()
    stim_displayed = True
    response_captured = False
    sub_resp = None
    rt = None
    probe1 = 0
    previous_resp = 0
    kb.clock.reset()
    timestamp = None
    for frame_n in range(total_probe1_frames):
        if 0 <= frame_n <= probe1_frames and response_captured == False:
            stim.draw()
            probe1_resp1.draw()
            probe1_resp2.draw()
            vertical_line.draw()
            if frame_n == 0:
                stim_onset = global_clock.getTime() * 1000
        keys = kb.getKeys(['left', 'right'])
        for key in keys:
            rt = key.rt*1000
            timestamp = float(rt) + float(stim_onset)
            sub_resp = key.name
            probe1 = 1 if key.name == 'left' else 0
            previous_resp = probe1
            # if 0 <= frame_n <= probe1_frames:
            stim_displayed = False
            response_captured = True
        window.flip()
        if response_captured:
            break
    end_time = trial_clock.getTime() * 1000          
    add_trial_data(trial, stim_onset, end_time)
    thisExp.addData('rt', rt)
    thisExp.addData('timestamp', timestamp)
    thisExp.addData('response', sub_resp)
    thisExp.addData('probe1', probe1)
    
    return previous_resp
    
def run_probe2_trial(trial, trial_clock, previous_resp, total_probe2_frames, probe2_frames):
    '''
    Handles the logic for a probe2 trial, including displaying stimuli and recording responses.
    
    trial (dict): a dictionary containing trial information.
    trial_clock (function): a Clock function from Psychopy.
    previous_resp (int): calls run_probe1_trial to return previous probe1 response and determines stimulus in probe2.
    tota2_probe1_frames (int): the total number of frames for the probe2 display and response.
    probe2_frames (int): number of frames that the probe2 stimulus is displayed.
    '''
    trial_clock.reset()
    stim_displayed = True
    sub_resp = None
    probe2 = None
    rt = None
    kb.clock.reset()
    timestamp = None
    if previous_resp == 1:
        for frame_n in range(total_probe2_frames):
            if 0 <= frame_n < probe2_frames and stim_displayed:
                stim.draw()
                probe2_resp1.draw()
                probe2_resp2.draw()
                vertical_line.draw()
                if frame_n == 0:
                    stim_onset = global_clock.getTime() * 1000 
            keys = kb.getKeys(['left', 'right'])
            for key in keys:
                rt = key.rt*1000
                sub_resp = key.name
                timestamp = float(rt) + float(stim_onset)
                probe2 = 1 if key.name == trial['probe2CorrAns'] else 0 # 1 = correct, 0 = incorrect
                if 0 <= frame_n < probe2_frames:
                    stim_displayed = False
            window.flip()
        end_time = trial_clock.getTime() * 1000 
        add_trial_data(trial, stim_onset, end_time)
        thisExp.addData('response', sub_resp)
        thisExp.addData('rt', rt)
        thisExp.addData('timestamp', timestamp)
        thisExp.addData('probe2', probe2)
    elif previous_resp == 0:
        stim.setText("Kde se vaše mysl nacházela, když nebyla zaměřena na úkol?")
        for frame_n in range(total_probe2_frames):
            if 0 <= frame_n < probe2_frames and stim_displayed:
                stim.draw()
                probe2_resp3.draw()
                probe2_resp4.draw()
                vertical_line.draw()
                if frame_n == 0:
                    stim_onset = global_clock.getTime() * 1000
            keys = kb.getKeys(['left', 'right'])
            for key in keys:
                rt = key.rt*1000
                timestamp = float(rt) + float(stim_onset)
                sub_resp = key.name 
                probe2 = 2 if key.name == 'left' else 3 # 2 = external distraction, 3 = daydreaming
                if 0 <= frame_n < probe2_frames:
                    stim_displayed = False
            window.flip()
        end_time = trial_clock.getTime() * 1000 
        add_trial_data(trial, stim_onset, end_time)
        thisExp.addData('stimulus', "Kde se vaše mysl nacházela, když nebyla zaměřena na úkol?")
        thisExp.addData('response', sub_resp)
        thisExp.addData('rt', rt)
        thisExp.addData('timestamp', timestamp)
        thisExp.addData('probe2', probe2)
        
def quit_experiment():
    '''
    Aborts the run immediately from anywhere, saving whatever data has been
    collected so far. Registered as a global key handler, so it fires from
    within window.flip() regardless of where execution currently is.
    '''
    try:
        thisExp.saveAsWideText(filename + '.csv', delim='auto')
        thisExp.saveAsPickle(filename)
        logging.flush()
    except Exception:
        pass
    thisExp.abort()
    window.close()
    core.quit()

# Abort the run at any time with <escape>
event.globalKeys.add(key='escape', func=quit_experiment)

# Sets condition files for either practice or real experiment
if exp_info['practice'] == 'No':
    block_files = [
    'Block1.xlsx'
]
else:
    block_files = ['BlockPractice.xlsx']

# Main Experiment Loop   
for block in block_files:
    if block_count == 1:
        display_instructions(PRACTICE_INSTRUCTIONS)
    trials = initialize_trial_handler(block)
    display_break(5, block_count)
    display_blank()
    trial_count = 1
    previous_resp = 0
    trial_clock = core.Clock()
    # global_clock.reset()
    for trial in trials:
        stimulus = trial['stimulus']
        stim.setText(stimulus)
        kb.clearEvents(eventType='keyboard')
        if trial['trialType'] == 'Non-target' or trial['trialType'] == 'Target':
            run_number_trial(trial, trial_clock, total_num_frames, num_frames)
        elif trial['trialType'] == 'Probe 1':
            previous_resp = run_probe1_trial(trial, trial_clock, total_probe1_frames, probe1_frames)
        elif trial['trialType'] == 'Probe 2':
            run_probe2_trial(trial, trial_clock, previous_resp, total_probe2_frames, probe2_frames)
        trial_count+=1
        thisExp.nextEntry()
        if 'escape' in kb.getKeys():
            window.close()
            core.quit()
    block_count+=1
display_complete()

window.flip()        
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()

thisExp.abort()
window.close()
core.quit()        
            
                
