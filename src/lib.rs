use js_sys::wasm_bindgen::JsValue;
use screeps::{RoomTerrain, Terrain};
use smallvec::{SmallVec, smallvec};

use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;

const TERRAIN_MASK_WALL: Terrain = Terrain::Wall;

const UNWALKABLE: i16 = -10;
const RANGE_MODIFIER: i16 = 1; // this parameter sets the scaling of weights to prefer walls closer protection bounds
const RANGE_PADDING: i16 = 3; // max range to reduce weighting; RANGE_MODIFIER * RANGE_PADDING must be < PROTECTED
const NORMAL: i16 = 0;
const PROTECTED: i16 = 10;
const CANNOT_BUILD: i16 = 20;
const EXIT: i16 = 30;

struct Edge {
	capacity: i16,
	flow: i16,
	res_edge: usize,
	to: usize
}

#[derive(Serialize, Deserialize)]
pub struct Rectangle {
    pub x1: usize,
    pub y1: usize,
    pub x2: usize,
    pub y2: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Coord {
    pub x: i16,
    pub y: i16,
}

struct MinCutEdge {
    to: usize,
    unreachable: usize,
}

struct Graph {
    total_vertices: usize,
    level: Vec<i32>,
    edges: Vec<SmallVec<[Edge; 4]>>,
}

impl Graph {
    fn new(total_vertices: usize) -> Graph {
        let mut edges = Vec::with_capacity(total_vertices);
        for _ in 0..total_vertices {
            edges.push(SmallVec::new());
        }
        Graph {
            total_vertices,
            level: vec![0; total_vertices],
            edges,
        }
    }

    fn new_edge(&mut self, from: usize, to: usize, capacity: i16) {
        let to_edge_len: usize = self.edges[to].len();

        self.edges[from].push(Edge {
            to,
            res_edge: to_edge_len,
            capacity,
            flow: 0,
        });

        // reverse Edge for Residual Graph
        let from_edge_len: usize = self.edges[from].len() -1;
        self.edges[to].push(Edge {
            to: from,
            res_edge: from_edge_len,
            capacity: 0,
            flow: 0,
        });
    }

    fn create_level_graph(&mut self, from: usize, to: usize) -> bool {
        if to >= self.total_vertices {
            return false;
        }
        self.level.fill(-1);
        self.level[from] = 0;

        let mut q: Vec<usize> = vec![from];
        while let Some(u) = q.pop() {
            for edge in &self.edges[u] {
                if self.level[edge.to] < 0 && edge.flow < edge.capacity {
                    self.level[edge.to] = self.level[u] + 1;
                    q.push(edge.to);
                }
            }
        }
        self.level[to] >= 0
    }
    fn calc_flow(&mut self, start: usize, end: usize, target_flow: i16, count: &mut Vec<usize>) -> i16 {
        if start == end {
            return target_flow;
        }

        let mut edge_flow: i16;
        while count[start] < self.edges[start].len() {
            let edge_index = count[start]; // Save the index before mutable borrow
            let edge_to = self.edges[start][edge_index].to; // Save needed values before mutable borrow
            let edge_res_edge = self.edges[start][edge_index].res_edge;
            let edge_capacity = self.edges[start][edge_index].capacity;
            edge_flow = self.edges[start][edge_index].flow;
    
            if self.level[edge_to] == self.level[start] + 1 && edge_flow < edge_capacity {
                let flow_till_here = target_flow.min(edge_capacity - edge_flow);
                // Perform the recursive call without any references to self
                let flow_to_t = self.calc_flow(edge_to, end, flow_till_here, count);
                if flow_to_t > 0 {
                    // Access `self` again after the mutable borrow for recursion is done
                    self.edges[start][edge_index].flow += flow_to_t;
                    self.edges[edge_to][edge_res_edge].flow -= flow_to_t;
                    return flow_to_t;
                }
            }
            count[start] += 1;
        }
        0
    }
    
    fn get_min_cut(&mut self, from: usize) -> Vec<usize> {
        let mut e_in_cut: Vec<MinCutEdge> = Vec::new();
        self.level.fill(-1);
        self.level[from] = 1;

        let mut q: Vec<usize> = Vec::new();
        q.push(from);

        while let Some(u) = q.pop() {
            for edge in &self.edges[u] {
                if edge.flow < edge.capacity {
                    if self.level[edge.to] < 1 {
                        self.level[edge.to] = 1;
                        q.push(edge.to);
                    }
                }
                if edge.flow == edge.capacity && edge.capacity > 0 {
                    e_in_cut.push(MinCutEdge { to: edge.to, unreachable: u });
                }
            }
        }

        let mut min_cut = Vec::new();
        for cut_edge in e_in_cut {
            if self.level[cut_edge.to] == -1 {
                min_cut.push(cut_edge.unreachable);
            }
        }
        min_cut
    }

    fn calc_min_cut(&mut self, source: usize, sink: usize) -> i16 {
        if source == sink {
            return -1;
        }

        let mut ret: i16 = 0;
        let mut count: Vec<usize>;
        let mut flow: i16;

        while self.create_level_graph(source, sink) {
            count = vec![0; self.total_vertices + 1];
            loop {
                flow = self.calc_flow(source, sink, i16::MAX, &mut count);
                if flow > 0 {
                    ret += flow;
                } else {
                    break;
                }
            }
        }
        ret
    }
}


fn get_2d_array(bounds: &Rectangle, terrain: &RoomTerrain) -> [[i16; 50]; 50] {
    let mut room_2d: [[i16; 50]; 50] = [[NORMAL; 50]; 50];  // Initialize the 2D array

    for x in bounds.x1..=bounds.x2 {
        for y in bounds.y1..=bounds.y2 {
            if terrain.get(x as u8, y as u8) == TERRAIN_MASK_WALL {
                room_2d[x][y] = UNWALKABLE;
            } else if x == bounds.x1 || y == bounds.y1 || x == bounds.x2 || y == bounds.y2 {
                room_2d[x][y] = EXIT;
            }
        }
    }

    // Marks tiles as unbuildable if they are proximate to exits
    for y in bounds.y1 + 1..=bounds.y2 - 1 {
        if room_2d[bounds.x1][y] == EXIT {
            let x_bound_limit = bounds.x1+1;
            //for &dy in &[-1, 0, 1] {
                //-1
                if room_2d[x_bound_limit][y -1] != UNWALKABLE {
                    room_2d[x_bound_limit][y -1] = CANNOT_BUILD;
                }
                //+0
                if room_2d[x_bound_limit][y] != UNWALKABLE {
                    room_2d[x_bound_limit][y] = CANNOT_BUILD;
                }
                //+1
                if room_2d[x_bound_limit][y + 1] != UNWALKABLE {
                    room_2d[x_bound_limit][y + 1] = CANNOT_BUILD;
                }
            //}
        }
        if room_2d[bounds.x2][y] == EXIT {
            //for &dy in &[-1, 0, 1] {
                let x_bound_limit = bounds.x2 -1;
                //-1
                if room_2d[x_bound_limit][y - 1] != UNWALKABLE {
                    room_2d[x_bound_limit][y - 1] = CANNOT_BUILD;
                }
                //+0
                if room_2d[x_bound_limit][y] != UNWALKABLE {
                    room_2d[x_bound_limit][y] = CANNOT_BUILD;
                }
                //+1
                if room_2d[x_bound_limit][y + 1] != UNWALKABLE {
                    room_2d[x_bound_limit][y + 1] = CANNOT_BUILD;
                }
            //}
        }
    }

    for x in bounds.x1 + 1..=bounds.x2 - 1 {
        if room_2d[x][bounds.y1] == EXIT {
            //for &dx in &[-1, 0, 1] {
                let y_bound_limit = bounds.y1+1;
                //-1
                if room_2d[x - 1][y_bound_limit] != UNWALKABLE {
                    room_2d[x - 1][y_bound_limit] = CANNOT_BUILD;
                }
                //+0
                if room_2d[x][y_bound_limit] != UNWALKABLE {
                    room_2d[x][y_bound_limit] = CANNOT_BUILD;
                }
                //+1
                if room_2d[x + 1][y_bound_limit] != UNWALKABLE {
                    room_2d[x + 1][y_bound_limit] = CANNOT_BUILD;
                }
            //}
        }
        if room_2d[x][bounds.y2] == EXIT {
            //for &dx in &[-1, 0, 1] {
                let y_bound_limit = bounds.y2 -1;
                //-1
                if room_2d[x - 1][y_bound_limit] != UNWALKABLE {
                    room_2d[x - 1][y_bound_limit] = CANNOT_BUILD;
                }
                //+0
                if room_2d[x][y_bound_limit] != UNWALKABLE {
                    room_2d[x][y_bound_limit] = CANNOT_BUILD;
                }
                //+1
                if room_2d[x + 1][y_bound_limit] != UNWALKABLE {
                    room_2d[x + 1][y_bound_limit] = CANNOT_BUILD;
                }
            //}
        }
    }

    room_2d
}


const SURR: [[i16; 2]; 8] = [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]];

fn create_graph(terrain: &RoomTerrain, to_protect: &Vec<Rectangle>,
    prefer_closer_barriers: bool,
    prefer_closer_barrier_limit: usize,
    bounds: &Rectangle) -> Graph {

    let mut room_array = get_2d_array(&bounds, terrain);
    for r in to_protect {
        for x in r.x1..=r.x2 {
            for y in r.y1..=r.y2 {
                if x == r.x1 || x == r.x2 || y == r.y1 || y == r.y2 {
                    if room_array[x][y] == NORMAL {
                        room_array[x][y] = PROTECTED;
                    }
                } else {
                    room_array[x][y] = UNWALKABLE;
                }
            }
        }
    }
        
	// Preferentially weight closer tiles
	if prefer_closer_barriers {
        // Take only the first `prefer_closer_barrier_limit` elements of `to_protect`
        for r in to_protect.iter().take(prefer_closer_barrier_limit) {
            let xmin = std::cmp::max(r.x1 as isize - RANGE_PADDING as isize, 0) as usize;
            let xmax = std::cmp::min(r.x2 as isize + RANGE_PADDING as isize, 49) as usize;
            let ymin = std::cmp::max(r.y1 as isize - RANGE_PADDING as isize, 0) as usize;
            let ymax = std::cmp::min(r.y2 as isize + RANGE_PADDING as isize, 49) as usize;

            for x in xmin..=xmax {
                for y in ymin..=ymax {
                    if room_array[x][y] >= NORMAL && room_array[x][y] < PROTECTED {
                        let x1range = std::cmp::max(r.x1 as isize - x as isize, 0) as usize;
                        let x2range = std::cmp::max(x as isize - r.x2 as isize, 0) as usize;
                        let y1range = std::cmp::max(r.y1 as isize - y as isize, 0) as usize;
                        let y2range = std::cmp::max(y as isize - r.y2 as isize, 0) as usize;
                        let range_to_border = std::cmp::max(std::cmp::max(x1range, x2range), std::cmp::max(y1range, y2range));
                        let modified_weight = NORMAL + RANGE_MODIFIER * (RANGE_PADDING as isize - range_to_border as isize) as i16;
                        room_array[x][y] = std::cmp::max(room_array[x][y], modified_weight);
                    }
                }
            }
        }
    }
	// possible 2*50*50 +2 (st) Vertices (Walls etc set to unused later)
	let mut g = Graph::new(2 * 50 * 50 + 2);

	let source = 2 * 50 * 50;
	let sink = 2 * 50 * 50 + 1;
	let base_capacity = 10;
	let modify_weight = if prefer_closer_barriers { 1 } else { 0} ;
    for x in bounds.x1+1..=bounds.x2-1 {
        for y in bounds.y1+1..=bounds.y2-1 {
			let top = y * 50 + x;
			let bot = top + 2500;
			if room_array[x][y] >= NORMAL && room_array[x][y] <= PROTECTED {
				if room_array[x][y] >= NORMAL && room_array[x][y] < PROTECTED {
					g.new_edge(top, bot, base_capacity - modify_weight * room_array[x][y]); // add surplus weighting
				} else if room_array[x][y] == PROTECTED { // connect this to the source
					g.new_edge(source, top, i16::MAX);
					g.new_edge(top, bot, base_capacity - modify_weight * RANGE_PADDING * RANGE_MODIFIER);
				}
				for i in 0..=7 { // attach adjacent edges
					let dx = x + SURR[i][0] as usize;
					let dy = y + SURR[i][1] as usize;
					if {room_array[dx][dy] >= NORMAL && room_array[dx][dy] < PROTECTED }
						|| room_array[dx][dy] == CANNOT_BUILD {
						g.new_edge(bot, dy * 50 + dx, i16::MAX);
					}
				}
			} else if room_array[x][y] == CANNOT_BUILD { // near Exit
				g.new_edge(top, sink, i16::MAX);
			}
		}
	} // graph finished
	g
}

#[wasm_bindgen]
pub fn get_cut_tiles(terrain: RoomTerrain, to_protect: JsValue, prefer_closer_barriers: bool, prefer_closer_barrier_limit: usize, bounds: JsValue) -> Result<JsValue, JsValue> {
    // Deserialize JsValue back to Rust types
    //let terrain: RoomTerrain = serde_wasm_bindgen::from_value(terrain)?;
    let to_protect: Vec<Rectangle> = serde_wasm_bindgen::from_value(to_protect)?;
    let bounds: Rectangle = serde_wasm_bindgen::from_value(bounds)?;

    let mut graph = create_graph(&terrain, &to_protect, prefer_closer_barriers, prefer_closer_barrier_limit, &bounds);

    let source = 2 * 50 * 50; // Position Source / Sink in Room-Graph
    let sink = 2 * 50 * 50 + 1;
    let count = graph.calc_min_cut(source, sink);
    let mut positions: Vec<Coord> = Vec::new();
    if count > 0 {
        let cut_vertices = graph.get_min_cut(source);
        for v in cut_vertices {
            let x = (v % 50) as i16;
            let y = (v / 50) as i16;
            positions.push(Coord { x, y });
        }
    }

    // Convert Rust Vec<Coord> back to JsValue to return
    Ok(serde_wasm_bindgen::to_value(&positions)?)
}