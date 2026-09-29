import React, { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useGameStore } from './GameStore'

export function WobbleStack({ count = 10, spacing = 0.15 }) {
  const meshRef = useRef()
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const combo = useGameStore((s) => s.combo)
  const motionIntensity = 0.05 + (combo * 0.01)

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    for (let i = 0; i < count; i++) {
      dummy.position.set(0, 0, -i * spacing)
      const phaseLag = i * 0.04
      const localT = t - phaseLag
      const zRotation = Math.sin(localT * Math.PI) * motionIntensity
      dummy.rotation.set(0, 0, zRotation)
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)
    }
    meshRef.current.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <planeGeometry args={[2, 3]} />
      <meshStandardMaterial color="#ff4081" transparent opacity={0.8} side={THREE.DoubleSide} />
    </instancedMesh>
  )
}
