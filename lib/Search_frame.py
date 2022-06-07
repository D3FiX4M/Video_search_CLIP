from PIL import Image


def find_top_frames(frame_name, text_features, image_features, top=1, power=100):
    proba = (power * text_features @ image_features.T).softmax(dim=-1)[0]

    items, idx = proba.topk(top)

    frame_idx = [frame_name[idx] for idx in idx.tolist()]

    result = {item[0]: item[1] for item in zip(frame_idx, items.tolist())}

    sorted_result = dict(sorted(result.items(), key=lambda x: x[0]))

    return sorted_result


def show_top_frames(items, path):
    for item in items:
        frame = Image.open(path + item)
        frame.thumbnail((1024, 1024))
        frame.show()
