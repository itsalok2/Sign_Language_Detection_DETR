import json 
import os 

def get_classes():
    try:
        with open('src/config.json') as f:
            config = json.load(f)

        classes = config['classes']
        assert len(classes)>0, 'You need to specify classes inside of the config.json file.'
        return classes 
    except Exception as e:
        return f'something wert wrong loading your config file: {e}'
    


def get_colors():
    try:
        with open('src/config.json') as f:
            config = json.load(f)

        classes = config['classes']
        colors = config['colors']

        assert len(colors)>0, 'You need to specify colors in RGB inside of config.json e.g. {"colors":[[131, 193, 103], [240, 172, 95]]}'
        assert len(classes) == len(colors), f'Please specify one color per class. You have {len(colors)} colours and {len(classes)} classes'
        return colors 
    except Exception as e:
        return f'something went wrong loading your config file: {e}'
    

if __name__== '__main__':
    classes = get_classes()
    print(classes)