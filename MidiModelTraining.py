import music21
import numpy
import keras
import glob
import pickle

#Data Prep
def getData():
    notes = []
    for file in glob.glob("midi_files/*.mid"):
        midi = music21.converter.parse(file)
        toParse = None

        #check for multi instrument tracks
        instruments = music21.instrument.partitionByInstrument(midi)
        if instruments:
            toParse = instruments.parts[0].recurse()
        else:
            toParse = midi.flat.notes
        
        '''
        Each time something is played it can be a note
        which is just a single thing like F5 or A5
        OR
        it can be a chord which is a set of multiple notes
        which can be F5, A5 and G4 
        so need to sort those accordingly
        '''
        for noteChords in toParse:
            if isinstance(noteChords, music21.note.Note):
                notes.append(str(noteChords.pitch)) # pitch gives name of note
            elif isinstance(noteChords, music21.chord.Chord):
                notes.append('.'.join(str(Chordnotes) for Chordnotes in noteChords.normalOrder))
    with open('data/music', 'wb') as filepath:
        pickle.dump(notes, filepath)
    print("data extracted\n")
    return notes

#Prepare data for training

def prepSequences(notes, noteVocab):
    #Prediction using previous x amount of notes
    sequenceLen = 50

    #Converting the notes to an int to allow for keras LSTM to use them
    #get all pitches in training data
    noteNames = sorted(set(item for item in notes))
    #making a dict to map notes to ints
    noteToInt = dict((note,number) for number, note in enumerate(noteNames))
    netIn = []
    netOut = []

    #create sets of input sequence and matching outputs
    #output will be the note immedietly after input sequence
    for i in range(0, len(notes) - sequenceLen, 1):
        seqIn = notes[i:i + sequenceLen]
        seqOut = notes[i + sequenceLen]

        netIn.append([noteToInt[char] for char in seqIn])
        netOut.append(noteToInt[seqOut])

    #reshape for input into LSTM
    nPatt = len(netIn)
    netIn = numpy.reshape(netIn, (nPatt, sequenceLen, 1))
    #normalize
    netIn = netIn / float(noteVocab)
    netOut = keras.utils.to_categorical(netOut)
    print("data preprocessing done\n")
    return (netIn, netOut)

#Training and creation
def createLSTM(netIn, noteVocab):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(
        512, 
        input_shape = (netIn.shape[1], netIn.shape[2]),
        return_sequences = True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(512, return_sequences = True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(512))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(noteVocab))
    model.add(keras.layers.Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy',
    optimizer = 'rmsprop')
    print("model done__________\n")
    return model

def training(model, netIn, netOut):
    fPath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
    checkPoint = keras.callbacks.ModelCheckpoint(fPath, 
    monitor = 'loss', verbose = 0, 
    save_best_only = True, mode = 'min')
    callbacks = [checkPoint]
    print("Training start\n")
    model.fit(netIn, netOut, epochs = 100, batch_size = 64, callbacks = callbacks)
    print("Training done")

def LSTMTrainer():
    notes = getData()
    noteVocab = len(set(notes))
    netIn, netOut = prepSequences(notes,noteVocab)
    model = createLSTM(netIn, noteVocab)

    training(model, netIn, netOut)

if __name__ == '__main__':
    LSTMTrainer()