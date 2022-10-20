def intersect_bboxes(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x_1, y_1, x_2, y_2 = bbox2
    tlx = max(x1, x_1)
    tly = max(y1, y_1)
    brx = min(x2, x_2)
    bry = min(y2, y_2)
    cond1, cond2 = (brx - tlx) > 0, (bry - tly) > 0

    if cond1 and cond2:
        return True

    return False