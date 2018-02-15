import QtQuick 2.7
import QtQuick.Window 2.2
import QtQuick.Controls 2.1
import QtQuick.Controls.Material 2.1
import PerceptronImagePainter 1.0

ApplicationWindow {
    Material.theme: Material.Dark
    Material.accent: Material.Purple
    title: "Testing"
    visible: true
    width: 400
    height: 250
    x: Screen.width / 2 + 1
    y: (Screen.height - height) / 2

    onClosing: {
        testGui.paused = true
    }

    Label {
        id: resArr
        anchors {
            horizontalCenter: parent.horizontalCenter
            top: parent.top
        }
        font.pixelSize: 10
        text: testGui.expected + "\n" + testGui.result
    }

    Label {
        anchors {
            left: parent.left
            top: imgBorder.top
        }
        text: "Expected: " + testGui.expectednum + "\nResult: " + testGui.resultnum
    }

    Rectangle {
        id: imgBorder
        width: 152
        height: 152
        anchors.centerIn: parent
        color: "black"
        ImagePainter {
            id: imgPainter
            anchors.centerIn: parent
            width: 150
            height: 150
            image: testGui.image
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
            testGui.nextStep()
        }
    }

    Label {
        anchors {
            horizontalCenter: parent.horizontalCenter
            verticalCenter: nxtStep.verticalCenter
        }
        text: "Step: " + (testGui.index + 1)
    }



    Button {
        anchors {
            left: parent.left
            bottom: parent.bottom
        }
        text: testGui.paused ? "Start" : "Pause"
        onClicked: {
            if(testGui.paused) {
                testGui.paused = false
            } else {
                testGui.paused = true
            }
        }
    }
}