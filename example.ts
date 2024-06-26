import 'fastestsmallesttextencoderdecoder-encodeinto/EncoderDecoderTogether.min.js';

/**
 * @property {number} x1 - Top left corner
 * @property {number} x1 - Top left corner
 * @property {number} x2 - Bottom right corner
 * @property {number} y2 - Bottom right corner
 */
export interface Rectangle {
	x1: number;
	y1: number;
	x2: number;
	y2: number;
}

const wasm_module = new WebAssembly.Module(require('mincut_Screeps_bg'))
import { initSync, get_cut_tiles } from '../wasm/mincut/mincut_Screeps.js'

export const wasm = initSync(wasm_module)

export function getCutTilesWasm(terrain: RoomTerrain, toProtect: Rectangle[],
	preferCloserBarriers     = true,
	preferCloserBarrierLimit = 4,
	bounds: Rectangle        = {x1: 0, y1: 0, x2: 49, y2: 49}) {
    const result = get_cut_tiles(terrain, toProtect, preferCloserBarriers, preferCloserBarrierLimit, bounds);
    return result;
}