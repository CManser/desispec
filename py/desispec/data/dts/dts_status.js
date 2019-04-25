$(function() {
    var Status = {
        // Raw data read from JSON file.
        raw: [],
        setRaw: function(data) { return this.raw = data; },
        // Ordered list of nights, most recent first.
        nights: [],
        startNight: function(night) {
            this.nights.push(night);
            return $("<div/>", {"id": ""+night, "class": "row"});
        },
        finishNight: function(night, buttons, b_rows, t, t_rows) {
            if (b_rows.length > 0) {
                t_rows.push("</table>");
                buttons.append(b_rows.join(""));
                t.append(t_rows.join(""));
                night.append(buttons);
                night.append(t);
                night.appendTo("#content");
            }
        },
        padExpid: function(expid) {
            var e = ("" + expid).split("");
            while (e.length < 8) e.unshift("0");
            return e.join("");
        },
        nightButton: function(n, role, success) {
            var color = success ? "btn-success" : "btn-danger";
            if (role == "show") {
                return "<button type=\"button\" class=\"btn " + color +
                    " btn-sm\" id=\"show" + n +
                    "\" style=\"display:inline;\" onclick=\"$('#t" + n +
                    "').css('display', 'block');$('#hide" + n +
                    "').css('display', 'inline');$('#show" + n +
                    "').css('display', 'none');\">Show</button>";
            } else {
                return "<button type=\"button\" class=\"btn " + color +
                    " btn-sm\" id=\"hide" + n +
                    "\" style=\"display:none;\" onclick=\"$('#t" + n +
                    "').css('display', 'none');$('#show" + n +
                    "').css('display', 'inline');$('#hide" + n +
                    "').css('display', 'none');\">Hide</button>";
            }
        },
        display: function() {
            $("#content").empty();
            var b_rows = [];
            var t_rows = [];
            var night, buttons, t;
            for (var k = 0; k < this.raw.length; k++) {
                var n = this.raw[k][0];
                if (this.nights.indexOf(n) == -1) {
                    //
                    // Finish previous night
                    //
                    this.finishNight(night, buttons, b_rows, t, t_rows);
                    //
                    // Start a new night
                    //
                    night = this.startNight(nights, n);
                    buttons = $("<div/>", {"class": "col-4"});
                    t = $("<div/>", {"class": "col-8"});
                    b_rows = ["<p id=\"p" + n + "\">Night " + n + "&nbsp;",
                              this.nightButton(n, "show", true),
                              this.nightButton(n, "hide", true),
                              "</p>"];
                    t_rows = ["<table id=\"t" + n + "\" style=\"display:none;\">"];
                }
                //
                // Add to existing night
                //
                var p = this.padExpid(this.raw[k][1]);
                // var s = this.raw[k][2];
                // if (exposureMeta.hasOwnProperty(p)) {
                //     // update
                // } else {
                //     // new exposure
                // }
                var c = this.raw[k][2] ? "bg-success" : "bg-danger";
                if (!this.raw[k][2]) {
                    b_rows[1] = this.nightButton(n, "show", false);
                    b_rows[2] = this.nightButton(n, "hide", false);
                }
                var l = this.raw[k][3].length > 0 ? " Last " + this.raw[k][3] + " exposure." : "";
                var r = "<tr id=\"e" + n + "/" + p +"\">" +
                        "<td>" + p + "</td>" +
                        "<td class=\"" + c + "\">" + l + "</td>" +
                        "</tr>"
                // var r = "<li class=\"" + c + "\" id=\"" +
                //         n + "/" + p + "\">" +
                //         p + l + "</li>";
                // console.log(r);
                t_rows.push(r);
            }
            //
            // Finish the final night
            //
            this.finishNight(night, buttons, b_rows, t, t_rows);
        }
    };
    $.getJSON("dts_status.json", {}, Status.setRaw).always(Status.display);
    return true;
});
