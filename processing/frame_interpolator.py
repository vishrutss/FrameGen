from models.rife_model import RIFEModel

def interpolate_frames(frames, target_fps=120, original_fps=30):
    model = RIFEModel()
    scale = target_fps // original_fps
    new_frames = []
    for i in range(len(frames) - 1):
        new_frames.append(frames[i])
        for _ in range(scale - 1):
            # TODO: call model.interpolate(frames[i], frames[i+1])
            interp = frames[i]  # placeholder
            new_frames.append(interp)
    new_frames.append(frames[-1])
    return new_frames
