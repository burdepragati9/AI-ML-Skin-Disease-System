import mediapipe as mp

print(f'MediaPipe version: {mp.__version__}')
print(f'Has solutions: {hasattr(mp, "solutions")}')
print(f'Has tasks: {hasattr(mp, "tasks")}')

if hasattr(mp, 'tasks'):
    print(f'Tasks available: {dir(mp.tasks)}')
