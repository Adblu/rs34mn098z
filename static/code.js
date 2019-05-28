var feature1_ ,feature2_, feature3_, feature4_, feature5_, feature6_, feature7_, feature8_, feature9_, feature10_, feature11_, feature12_;

$(document).ready(function(){
  
  feature1_ = document.getElementById("feature1");
  feature2_ = document.getElementById("feature2");
  feature3_ = document.getElementById("feature3");
  feature4_ = document.getElementById("feature4");
  feature5_ = document.getElementById("feature5");
  feature6_ = document.getElementById("feature6");
  feature7_ = document.getElementById("feature7");
  feature8_ = document.getElementById("feature8");
  feature9_ = document.getElementById("feature9");
  feature10_ = document.getElementById("feature10");
  feature11_ = document.getElementById("feature11");
  feature12_ = document.getElementById("feature12");

});

$(document).on('click','#submit',function(){
    var feature1 = feature1_.value;
    var feature2 = feature2_.value;
    var feature3 = feature3_.value;
    var feature4 = feature4_.value;
    var feature5 = feature5_.value;
    var feature6 = feature6_.value;
    var feature7 = feature7_.value;
    var feature8 = feature8_.value;
    var feature9 = feature9_.value;
    var feature10 = feature10_.value;
    var feature11 = feature11_.value;
    var feature12 = feature12_.value;

    if(feature1 == "" || feature2 == "" || feature3 == "" || feature4 == "" || feature5 == "" || feature6 == "" || feature7 == "" || feature8 == "" || feature9 == "" || feature10 == "" || feature11 == "" || feature12 == ""){
      alert("empty fields not allowed");
    }
    else{
      var requestURL = "http://127.0.0.1:5000/predict?feature1="+feature1+"&feature2="+feature2+"&feature3="+feature3+"&feature4="+feature4+"&feature5="+feature5+"&feature6="+feature6+"&feature7="+feature7+"&feature8="+feature8+"&feature9="+feature9+"&feature10="+feature10+"&feature11="+feature11+"&feature12="+feature12;

      $.getJSON(requestURL, function(data) {
        console.log(data); 
        prediction = data['json_key_for_the_prediction'];
      $(".result").html("Response: " + prediction+'.');
      });



    }
  });

