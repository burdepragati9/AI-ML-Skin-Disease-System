import mediapipe as mp

print(f'MediaPipe version: {mp.__version__}')
print(f'Has tasks: {hasattr(mp, "tasks")}')

if hasattr(mp, 'tasks'):
    print(f'Tasks modules: {dir(mp.tasks)}')
    if hasattr(mp.tasks, 'python'):
        print(f'Tasks.python modules: {dir(mp.tasks.python)}')
        if hasattr(mp.tasks.python, 'vision'):
            print(f'Tasks.python.vision modules: {dir(mp.tasks.python.vision)}')
