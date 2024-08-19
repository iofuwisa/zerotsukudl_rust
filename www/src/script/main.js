let dl
const js = import("./wasm/deep_learning.js");

js.then(js => {
    // console.log(js)
    js.fetch()
  });

let canvasField;

window.onload = function() {

    document.getElementById("guessResultField").textContent="-"

    canvasField = new CanvasField("mainCanvas");

    // mouse events
    document.getElementById("mainCanvas").onmousedown = function(event) {
        canvasField.onmousedown(event);
    };
    document.getElementById("mainCanvas").onmousemove = function(event) {
        canvasField.onmousemove(event);
    };
    document.getElementById("mainCanvas").onmouseup = function(event) {
        canvasField.onmouseup(event);
    };

    // key evens
    document.body.addEventListener('keydown',
    event => {
        canvasField.keydown(event);
    });
    
};

class CanvasField {
    constructor(elementId) {
        this.canvas = document.getElementById(elementId);
        this.canvasContext = this.canvas.getContext('2d');
        this.shapes = [];
        this.drawingShape = null;
    }

    setDrawingShape(shape) {
        if (this.drawingShape != null) {
            this.shapes.push(this.drawingShape);
        }
        this.drawingShape = shape;
    }

    // mouse events
    onmousedown(event) {
        this.setDrawingShape(new PencilShape());
        this.drawingShape.onmousedown(event);
        this.render();
    }

    onmousemove(event) {
        if(this.drawingShape != null) {
            this.drawingShape.onmousemove(event);
        }
        this.render();
    }

    onmouseup(event) {
        if(this.drawingShape != null) {
            this.drawingShape.onmouseup(event);
        }
        this.render();
        
    }

    // key events
    keydown(event) {
        switch (event.keyCode) {
            case 90: // z
                if (event.ctrlKey) {
                    if (this.drawingShape != null){
                        console.log("delete drawing");
                        this.drawingShape = null;
                    } else {
                        console.log("delete array");
                        this.shapes.pop();
                    }
                }
                break;
            case 13: // Enter
                document.getElementById("guessResultField").textContent="loading.."
                console.log(this.canvasContext.getImageData(0, 0, this.canvas.width, this.canvas.height));
                js.then(js => {
                    js.guess()
                    .then(res => {
                        document.getElementById("guessResultField").textContent=res
                      })
                  });
                break;
            default:
                break;
        }
        this.render();
    }

    render() {
        this.canvasContext.clearRect(0, 0, this.canvas.width, this.canvas.height);
        for (const shape of this.shapes) {
            shape.render(this.canvasContext);
        }
        if(this.drawingShape != null) {
            this.drawingShape.render(this.canvasContext);
        }
    }
}

class PencilShape {
    constructor() {
        this.drawing = false;
        this.trajectory = [];
    }

    onmousedown(event) {
        this.drawing = true;
        this.trajectory.push(new Position(event.offsetX, event.offsetY));
    }

    onmousemove(event) {
        if (this.drawing) {
            this.trajectory.push(new Position(event.offsetX, event.offsetY));
        }
    }

    onmouseup(event) {
        if (this.drawing) {
            this.trajectory.push(new Position(event.offsetX, event.offsetY));
            this.drawing = false;
        }
    }

    render(context) {
        for (let i=0; i<this.trajectory.length-1; i++) {
            context.beginPath();
            context.moveTo(this.trajectory[i].x, this.trajectory[i].y);
            context.lineTo(this.trajectory[i+1].x, this.trajectory[i+1].y);
            context.lineWidth = 8;
            context.stroke();
        }
        context.lineWidth = 1;
    }

} 

class Position {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}