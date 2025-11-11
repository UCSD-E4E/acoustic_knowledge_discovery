function _1(md){return(
md`
`
)}

function _chart(d3,data,invalidation)
{
  // Specify the dimensions of the chart.
  const width = 1000;
  const height = 1200;

  // Specify the color scale.
  const color = d3.scaleOrdinal(d3.schemeCategory10);

  // The force simulation mutates links and nodes, so create a copy
  // so that re-evaluating this cell produces the same result.
  //const existingLinks = data.links.map(d => ({...d}));
  //const links = data.links.map()
  const nodes = data.nodes.map(d => ({...d}));

  // const predictedLinks = [
  //   { source: "Time of Day_Morning", target: "Location Name_Sand Forest", value: 1, predicted: true, confidence: 0.9014 },
  //   { source: "Time of Day_Morning", target: "Location Name_Lion House", value: 1, predicted: true, confidence: 0.9008 },
  //   //{ source: "Bat", target: "4PM - 11:59PM", value: 1, predicted: true, confidence: 0.2450 }
  // ];

  // const links = [...data.links.map(d => ({ ...d })), ...predictedLinks];
  const links = [...data.links.map(d => ({ ...d })),];
  //store # of edges for each node so that you dynamically change node size
  const degreeMap = new Map();
  nodes.forEach(d => degreeMap.set(d.id, 0)); //initialize to 0
  // Count connections
  links.forEach(link => {
    const sourceId = typeof link.source === "object" ? link.source.id : link.source;
    const targetId = typeof link.target === "object" ? link.target.id : link.target;
    degreeMap.set(sourceId, degreeMap.get(sourceId) + 1);
    degreeMap.set(targetId, degreeMap.get(targetId) + 1);
  });
  const radiusScale = d3.scaleLinear()
    .domain(d3.extent(nodes, d => degreeMap.get(d.id)))  // [minDegree, maxDegree]
    .range([6, 20]);  // adjust size range of nodes. rn, 6 is smallest and 20 is largest
  const fontSizeScale = d3.scaleLinear()
    .domain(d3.extent(nodes, d => degreeMap.get(d.id))) // [minDegree, maxDegree]
    .range([8, 16]);  // you can tweak this range as desired

  console.log("Post link processing")

  // Create a simulation with several forces.
  const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id))
      .force("linkLabels", d3.forceLink(links).id(d => d.id))
      .force("charge", d3.forceManyBody().strength(-1000))
      .force("x", d3.forceX())
      .force("y", d3.forceY());

  // Create the SVG container.
  const svg = d3.create("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [-width / 2, -height / 2, width, height])
      .attr("style", "height: auto; max-width: 100%;")

  console.log("after svg")

  // Add a line for each link, and a circle for each node.
  const link = svg.append("g")
    //.attr("stroke", "#999") //to apply same color to entire group
    .attr("stroke-opacity", 0.6)
    .selectAll("line")
    .data(links)
    .join("line") //traits for individual links
      .attr("stroke", d => d.predicted === true ? "#0da87fff" : "#999")
      .attr("stroke-width", d => Math.sqrt(d.value))
      .attr("stroke-dasharray", d => ( d.predicted === true) ? "4,4" : "0") //if it is predicted edge, make it dashed. otherwise, it is solid

  const node = svg.append("g")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
      //.attr("r", 10)
      .attr("r", d => radiusScale(degreeMap.get(d.id)))
      .attr("fill", d => color(d.group))
    .on("mouseover", hoverNode)
    .on("mouseout", endHover)
    .on("click", listAllConnections);
  
  const linkLabels = svg.append("g")
    .selectAll("text")
    .data(links.filter(d => typeof d.weight !== "undefined"))
    .join("text")
      .attr("font-size", "12px")
      .attr("fill", "#0da87fff")
      .text(d => d.weight.toFixed(2));


  node.append("title")
      .text(d => d.id);

  const labels = svg.append("g")
      .attr("class", "label")
    .selectAll("text")
    .data(nodes)
    .enter().append("text")
      .attr("dx", 10)
      .attr("dy", ".35em")
	   //.attr("style", "font-size: 10;")
     .attr("font-size", d => fontSizeScale(degreeMap.get(d.id)) + "px")
      .text(function(d) { return d.id });

  console.log("pre legend")
  
  // Create a legend group in the top-left corner
  const legend = svg.append("g")
    .attr("transform", `translate(${width / 2 - 150}, ${-height / 2 + 20})`); 

  const uniqueGroups = Array.from(new Set(nodes.map(d => d.group)));
  let activeGroups = new Set(uniqueGroups); //initially, all groups are active/enabled
  console.log("activeGroups are ", activeGroups)

  function updateVisibility() {
    // show/hide nodes & labels based on group
    console.log("activeGroups are ", activeGroups)
    node.style("display", n => activeGroups.has(n.group) ? null : "none");
    labels.style("display", n => activeGroups.has(n.group) ? null : "none");
    link.style("display", l =>
      (activeGroups.has(l.source.group) && activeGroups.has(l.target.group)) ? null : "none"
    );
    // hide confidence labels too if any of its endpoints are hidden
    if (linkLabels) {
      linkLabels.style("display", l =>
        (activeGroups.has(l.source.group) && activeGroups.has(l.target.group)) ? null : "none"
      );
    }
  }

  const legendItems = legend.selectAll("legend-item")
    .data(uniqueGroups)
    .enter()
    .append("g")
    .attr("class", "legend-item")
    .attr("transform", (d, i) => `translate(0, ${i * 20})`) // vertical spacing
    .style("cursor", "pointer")
    .on("click", function(event, d) {
      // Toggle selection
      if (activeGroups.has(d+1)) { //accomodate for the index being offset by 1
        activeGroups.delete(d+1); 
      }
      else { 
        activeGroups.add(d+1); 
      }
      updateVisibility();
      // optional: visually indicate active/inactive legend entry
      d3.select(this).select("circle")
        .attr("stroke", activeGroups.has(d+1) ? "none" : "#333")
        .attr("stroke-width", activeGroups.has(d+1) ? 0 : 5);
    });

  legendItems.append("circle")
    .attr("r", 6)
    .attr("fill", color);

  legendItems.append("text")
    .attr("x", 12)
    .attr("y", 4)
    .style("font-size", "12px")
    .text(d => `Group ${d}`);
  

  // Add a drag behavior.
  node.call(d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended));
  
  
   // Create a map from node ID to its neighbors
  const neighborMap = new Map();
  nodes.forEach(node => neighborMap.set(node.id, new Set()));
  links.forEach(link => {
    neighborMap.get(link.source.id || link.source).add(link.target.id || link.target);
    neighborMap.get(link.target.id || link.target).add(link.source.id || link.source);
  });


  
  // Set the position attributes of links and nodes each time the simulation ticks.
  simulation.on("tick", () => {
    link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

    node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

    labels
        .attr("x", function(d) { return d.x; })
        .attr("y", function(d) { return d.y; }); 
    
    linkLabels
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2);
  });

  // Reheat the simulation when drag starts, and fix the subject position.
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  // Update the subject (dragged node) position during drag.
  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function hoverNode(d){
    // Get neighbors of this node
    const neighbors = neighborMap.get(d.id);
    node
      .attr("opacity", n => neighbors.has(n.id) || n.id === d.id ? 1 : 0.2) //keep node and its neighbors hella opaque and fade out teh other ndoes
  
    labels
      .attr("opacity", n => neighbors.has(n.id) || n.id === d.id ? 1 : 0.2);

    // Highlight connected links
    link
      .attr("stroke-width", l =>
        (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id ? 2.5 : 1)
      .attr("stroke-opacity", l =>
        (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id ? 1 : 0.1);
    
  }

  function endHover(){
    //return size and opacity of all nodes and links to the original
    node
      .attr("opacity", 1);
    link
      .attr("stroke-width", 1)
      .attr("stroke-opacity", 1);
    labels
      .attr("opacity", 1);
  }

  function listAllConnections(d){
    const nodeId = d.id;
    const neighbors = neighborMap.get(nodeId);
    const sidebar = document.getElementById("sidebar");

    // Clear the sidebar
    sidebar.innerHTML = `<h3>${nodeId}</h3>`;

    if (!neighbors || neighbors.size === 0) {
      sidebar.innerHTML += `<p><em>No neighbors</em></p>`;
      return;
    }

    // Add a list of neighbors
    const ul = document.createElement("ul");
    Array.from(neighbors).forEach(nId => {
      const neighborNode = nodes.find(n => n.id === nId);
      const group = neighborNode?.group ?? 0;
      const colorStr = color(group);

      const li = document.createElement("li");
      li.textContent = nId;
      li.style.color = colorStr;
      li.style.fontWeight = "bold";

      ul.appendChild(li);
    });
    console.log(ul)
    sidebar.appendChild(ul);
  }

  // Restore the target alpha so the simulation cools after dragging ends.
  // Unfix the subject position now that it’s no longer being dragged.
  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  // When this cell is re-run, stop the previous simulation. (This doesn’t
  // really matter since the target alpha is zero and the simulation will
  // stop naturally, but it’s a good practice.)
  invalidation.then(() => simulation.stop());
  try {
    return svg.node();
  } catch (err) {
    console.error("Chart rendering failed:", err);
    throw err;
  }

}


function _color(d3){return(
d3.scaleOrdinal(d3.schemeCategory10)
)}

function _data(FileAttachment){return(
FileAttachment("graph_representation.json").json()
)}

function _d3(require){return(
require("d3@5")
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  function toString() { return this.url; }
  const fileAttachments = new Map([
    //["graph_representation.json", {url: new URL("./files/ebfa147f7218900de2e35380a53efbb7920c8d8d10bd70e45932efc56712293a937c8d4ffc73b604b2d5963fc6f2684ef36e9c82a16e5f80960805f185096a11.json", import.meta.url), mimeType: "application/json", toString}]
    ["graph_representation.json", {url: new URL("./graph_representation.json", import.meta.url), mimeType: "application/json", toString}]
  ]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer()).define(["md"], _1);
  main.variable(observer("chart")).define("chart", ["d3","data","invalidation"], _chart);
  main.variable(observer("color")).define("color", ["d3"], _color);
  main.variable(observer("data")).define("data", ["FileAttachment"], _data);
  main.variable(observer("d3")).define("d3", ["require"], _d3);
  return main;
}