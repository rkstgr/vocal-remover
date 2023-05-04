from mdx import SeperateMDX

# main cli entry point

def process(input_paths):
    """Start the conversion for all the given mp3 and wav files
    
    input_paths: list of paths to the audio files
    """
    for path in input_paths:
        data = construct_data(path)
        model = SeperateMDX(model, data)        

def main():
    pass


if __name__ == '__main__':
    main()