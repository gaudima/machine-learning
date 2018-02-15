import QtQuick 2.7
import QtQuick.Window 2.2
import QtQuick.Controls 2.1
import QtQuick.Controls.Material 2.1
import PerceptronImagePainter 1.0

ApplicationWindow {
    Material.theme: Material.Dark
    Material.accent: Material.Purple
    title: "Training"
    visible: true
    width: 400
    height: 250
    x: Screen.width / 2 - width - 1
    y: (Screen.height - height) / 2

    onClosing: {
        trainGui.paused = true
    }

    Label {
        anchors {
            horizontalCenter: parent.horizontalCenter
            top: parent.top
        }
        font.pixelSize: 10
        text: trainGui.expected + "\n" + trainGui.result
    }

    Rectangle {
        width: 152
        height: 152
        anchors.centerIn: parent
        color: "black"
        ImagePainter {
            id: imgPainter
            anchors.centerIn: parent
            width: 150
            height: 150
            image: trainGui.image
        }
    }
    Button {
        id: nxtStep
        anchors {
            right: parent.right
            bottom: parent.bottom
        }
        text: "Next Step"
        onClicked: {
            trainGui.nextStep()
        }
    }

    Label {
        anchors {
            horizontalCenter: parent.horizontalCenter
            verticalCenter: nxtStep.verticalCenter
        }
        text: "Step: " + (trainGui.index + 1)
    }

    Button {
        anchors {
            left: parent.left
            bottom: parent.bottom
        }
        text: trainGui.paused ? "Start" : "Pause"
        onClicked: {
            if(trainGui.paused) {
                trainGui.paused = false
            } else {
                trainGui.paused = true
            }
        }
    }

    TestWindow {
        id: testWindow
        visible: true
    }
}