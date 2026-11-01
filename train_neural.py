from neural_brain import train_neural

if __name__ == '__main__':
    result = train_neural()
    print('NYXVAULT neural training complete')
    print('Characters:', result['characters'])
    print('Vocabulary:', result['vocabulary'])
    print('Epochs:', result['epochs'])
    print('Loss:', result['loss'])
