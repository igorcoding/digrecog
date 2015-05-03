function erase($canvas, diggroupClassname) {
    var canvas = $canvas[0];
    var ctx = canvas.getContext("2d");
    var w = canvas.width;
    var h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgb(255,255,255)";
    ctx.fillRect(0, 0, w, h);
    $(diggroupClassname).removeClass('active');
}

function canvasWork($canvas) {
    var canvas = $canvas[0];
    var ctx = canvas.getContext("2d");
    var mousePressed = false;

    $canvas.mousedown(function (e) {
        mousePressed = true;
        draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
        //draw(e.pageX, e.pageY, false);
    });

    $canvas.mousemove(function (e) {
        if (mousePressed) {
            draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
            //draw(e.pageX , e.pageY , true);
        }
    });

    $canvas.mouseup(function (e) {
        mousePressed = false;
    });

	$canvas.mouseleave(function (e) {
        mousePressed = false;
    });

    var lastX = 0,
        currX = 0,
        lastY = 0,
        currY = 0;

    var lineColor = "black";
    var lineWidth = 25;

    function draw(x, y, isDown) {
        if (isDown) {
            ctx.beginPath();
            ctx.lineJoin = "round";
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = lineWidth;
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.closePath();
            ctx.stroke();
          }
          lastX = x; lastY = y;
    }
}

function getBase64Canvas($canvas, removeHeader) {
    var canvas = $canvas[0];
    var dataUrl = canvas.toDataURL('image/jpeg');
    console.log(dataUrl);
    if (removeHeader) {
        return removeHeaderBase64(dataUrl);
    } else {
        return dataUrl;
    }
}

function removeHeaderBase64(dataUrl) {
    return dataUrl.replace('data:image/jpeg;base64,', '');
}

function recogniseImage($canvas, $label) {
    var base64Img = getBase64Canvas($canvas, true);
    $.ajax({
        url: '/recognise/',
        method: 'post',
        data: {
            img_data: base64Img
        }
    })
        .done(function(data) {
            console.log(data);
            console.log(JSON.stringify(data.activation));
            var max = -1;
            var max_act = -1;
            for(var i = 0; i < data.activation.length; ++i) {
                if (data.activation[i] > max) {
                    max = data.activation[i];
                    max_act = i;
                }
            }
            console.log(max_act, max);
            $label.text(max_act);
        })
        .error(function(data) {
            toastr.error("Recognition failed");
            console.error(data);
        });
}

function transformImage($canvas, label) {
    var img = getBase64Canvas($canvas);
    return [img, label];
}

function addToDataset($dataset, d) {
    var $d = $('<div class="dataset-entry"> <img class="digit" src=""/> <div class="dr-label"></div </div>');
    $d.find('.digit')[0].src = d[0];
    $d.find('.dr-label').text(d[1]);
    $dataset.append($d);
}

function train(dataset) {
    $.ajax({
        url: '/train/',
        method: 'post',
        data: {
            dataset: JSON.stringify(dataset)
        }
    })
        .done(function(data) {
            toastr.success("Trained neural network successfully");
            console.log(data);
        })
        .error(function(data) {
            toastr.error("Training failed");
            console.error(data)
        });
}

jQuery(document).ready(function($) {
    var recogniser = $('#recogniser');
    var $canvas = recogniser.find('#dig_canvas');
    var $recogniseButton = recogniser.find('#recognise_btn');
    var $eraseButton = recogniser.find('#erase_btn');
    var $addToDatasetBtn = recogniser.find('#add_to_dataset_btn');
    var $label = recogniser.find('#label_field');
    var diggroupClassname = ".btn-group > .btn";
    var $diggroup = $(diggroupClassname);

    var $dataset = $('#dataset');
    var $trainButton = $('#train_btn');

    erase($canvas, diggroupClassname);

    var dataset = [];

    var num = null;
    $diggroup.on("click", function(){
        num = +this.innerHTML;
    });

    $eraseButton.on('click', function() {
        erase($canvas, diggroupClassname);
    });

    $addToDatasetBtn.on('click', function() {
        var d = transformImage($canvas, num);
        addToDataset($dataset, d);
        d[0] = removeHeaderBase64(d[0]);
        dataset.push(d);
        erase($canvas, diggroupClassname);
        console.log(dataset);
    });

    canvasWork($canvas, $eraseButton);
    $recogniseButton.on('click', function() {
        recogniseImage($canvas, $label);
    });

    $trainButton.on('click', function() {
        train(dataset);
    });


    toastr.options = toastr.options = {
      "closeButton": false,
      "debug": false,
      "newestOnTop": true,
      "progressBar": false,
      "positionClass": "toast-top-right",
      "preventDuplicates": true,
      "onclick": null,
      "showDuration": "300",
      "hideDuration": "1000",
      "timeOut": "5000",
      "extendedTimeOut": "1000",
      "showEasing": "swing",
      "hideEasing": "linear",
      "showMethod": "fadeIn",
      "hideMethod": "fadeOut"
    };

});