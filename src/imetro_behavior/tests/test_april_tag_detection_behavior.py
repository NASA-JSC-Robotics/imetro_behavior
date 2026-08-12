import pytest
from py_trees.common import Status
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from imetro_behavior.apriltag_detection_behavior import DetectAprilTagLocal


class MockNode:
    """Mock ROS node to handle logging during unit tests."""
    class MockLogger:
        def info(self, msg): pass
        def error(self, msg): pass
        def warn(self, msg): pass

    def get_logger(self):
        return self.MockLogger()


@pytest.fixture
def apriltag_behavior():
    """Fixture to initialize the DetectAprilTagLocal behavior and register its ports."""
    behavior = DetectAprilTagLocal(name="TestAprilTagBehavior")
    behavior.setup_ports()
    behavior.setup(node=MockNode())
    return behavior


def test_initial_state(apriltag_behavior):
    """Checks that the behavior initializes with the expected default variables and states."""
    assert apriltag_behavior.is_running is False
    assert apriltag_behavior.has_finished is False
    assert apriltag_behavior.success is False
    assert apriltag_behavior.attempt_count == 0


def test_update_missing_inputs(apriltag_behavior, monkeypatch):
    """Checks that the behavior handles missing blackboard inputs gracefully by returning RUNNING instead of crashing."""
    apriltag_behavior.initialise()
    
    # Mock get_input to return None when inputs are missing
    monkeypatch.setattr(apriltag_behavior, "get_input", lambda key, default=None: None)

    status = apriltag_behavior.update()
    assert status == Status.RUNNING


def test_max_attempts_failure(apriltag_behavior, monkeypatch):
    """Checks that the behavior returns FAILURE once max_attempts are reached without success."""
    apriltag_behavior.initialise()
    apriltag_behavior.max_attempts = 3

    mock_image = Image()
    mock_camera_info = CameraInfo()
    
    # Mock get_input via monkeypatch to bypass blackboard entirely
    monkeypatch.setattr(apriltag_behavior, "get_input", lambda key, default=None: {
        "rgb_image": mock_image,
        "camera_info": mock_camera_info,
        "tag_id": 0,
        "tag_size": 0.073
    }[key])

    monkeypatch.setattr(apriltag_behavior.detector, "detect", lambda *args, **kwargs: [])
    monkeypatch.setattr(apriltag_behavior.bridge, "imgmsg_to_cv2", lambda *args, **kwargs: None)

    # Drive the ticks explicitly simulating thread execution loops
    for i in range(3):
        status = apriltag_behavior.update()
        if apriltag_behavior.is_running:
            if apriltag_behavior.thread is not None:
                apriltag_behavior.thread.join(timeout=2.0)
            status = apriltag_behavior.update()

        if i < 2:
            assert status == Status.RUNNING
        else:
            assert status == Status.FAILURE


def test_successful_detection(apriltag_behavior, monkeypatch):
    """Simulate a successful background thread run and ensure SUCCESS status and pose output."""
    apriltag_behavior.initialise()

    mock_image = Image()
    mock_image.header = Header(frame_id="camera_frame")
    mock_camera_info = CameraInfo()
    mock_camera_info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]

    # Mock get_input via monkeypatch to bypass blackboard entirely
    monkeypatch.setattr(apriltag_behavior, "get_input", lambda key, default=None: {
        "rgb_image": mock_image,
        "camera_info": mock_camera_info,
        "tag_id": 0,
        "tag_size": 0.073
    }[key])

    class MockTag:
        tag_id = 0
        pose_t = [[0.5], [1.0], [1.5]]
        pose_R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    monkeypatch.setattr(apriltag_behavior.detector, "detect", lambda *args, **kwargs: [MockTag()])
    monkeypatch.setattr(apriltag_behavior.bridge, "imgmsg_to_cv2", lambda *args, **kwargs: None)

    # First tick triggers thread start
    status1 = apriltag_behavior.update()
    assert status1 == Status.RUNNING
    assert apriltag_behavior.is_running is True

    # Wait for background thread execution to finish
    if apriltag_behavior.thread is not None:
        apriltag_behavior.thread.join(timeout=3.0)

    # Second tick processes completion and writes output port
    status2 = apriltag_behavior.update()
    assert status2 == Status.SUCCESS
    assert apriltag_behavior.success is True
    
    # Assert directly against the stored result pose attribute
    assert isinstance(apriltag_behavior.result_pose, PoseStamped)
    assert apriltag_behavior.result_pose.pose.position.x == 0.5
    assert apriltag_behavior.result_pose.pose.position.y == 1.0
    assert apriltag_behavior.result_pose.pose.position.z == 1.5