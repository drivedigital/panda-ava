import React from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useGameStore } from './GameStore'

export function EarthquakeCamera({ basePosition = new THREE.Vector3(0, 2, 8) }) {
  const { camera } = useThree()
  const shakeIntensity = useGameStore((s) => s.shakeIntensity) 

  useFrame(({ clock }) => {
    if (shakeIntensity <= 0) {
      camera.position.lerp(basePosition, 0.1)
      return
    }
    const t = clock.elapsedTime
    const p = ((t % 0.4) + 0.4) % 0.4 / 0.4 
    const dx = (Math.sin(p * Math.PI * 6) + Math.sin(p * Math.PI * 13)) * 0.15 * shakeIntensity
    const dy = (Math.cos(p * Math.PI * 7) + Math.sin(p * Math.PI * 11)) * 0.15 * shakeIntensity
    const dz = (Math.sin(p * Math.PI * 5) + Math.cos(p * Math.PI * 17)) * 0.15 * shakeIntensity
    camera.position.set(basePosition.x + dx, basePosition.y + dy, basePosition.z + dz)
  })
  return null
}
