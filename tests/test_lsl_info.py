from gaze_model import ScreenGeometry
from tracker import LSL_CHANNELS, make_lsl_info


def test_kinectgaze_info_has_labels_units_and_geometry():
    info = make_lsl_info(ScreenGeometry(cam_x_cm=-9.4))
    assert info.name() == "KinectGaze" and info.channel_count() == len(LSL_CHANNELS) == 10
    ch = info.desc().child("channels").child("channel")
    got = []
    while not ch.empty():
        got.append((ch.child_value("label"), ch.child_value("unit")))
        ch = ch.next_sibling("channel")
    assert got == LSL_CHANNELS
    assert got[4] == ("head_yaw", "deg") and got[7] == ("head_dist", "cm")
    assert float(info.desc().child("screen").child_value("camera_x_cm")) == -9.4
