import { useEffect, useMemo, useRef, useState } from "react";
import SpriteText from "three-spritetext";
import * as THREE from "three";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { getMemoryGraph } from "@/lib/api";
import type { GraphEdge, GraphNode } from "@/lib/schemas";
import { useUiStore } from "@/store/ui";

import { Panel } from "./Panel";

type ViewNode = GraphNode & { child_ids?: string[]; x?: number; y?: number; z?: number };
type ViewEdge = GraphEdge & { source: string; target: string };

const colors: Record<string, string> = {
  episodic: "#3b82f6",
  semantic: "#a855f7",
  topic: "#c08532",
  scope: "#6366f1",
  summary: "#8b5cf6",
  memory: "#3b82f6"
};
const emptyGroups = new Set<string>();

export function MemoryGraphPanel() {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(emptyGroups);
  const [fullscreen, setFullscreen] = useState(false);
  const [selected, setSelected] = useState<ViewNode | null>(null);
  const [cameraZ, setCameraZ] = useState(980);
  const [fitNonce, setFitNonce] = useState(0);

  useEffect(() => {
    getMemoryGraph(120)
      .then((graph) => {
        setNodes(graph.nodes);
        setEdges(graph.edges);
      })
      .catch(() => {
        setNodes([]);
        setEdges([]);
      });
  }, []);

  const view = useMemo(
    () => graphViewData(nodes, edges, expandedGroups),
    [edges, expandedGroups, nodes]
  );

  function expandAll() {
    setExpandedGroups(new Set(view.groups.map((group) => group.id)));
  }

  function consolidateAll() {
    setExpandedGroups(new Set());
    setSelected(null);
  }

  return (
    <Panel title="Memory Graph" badge={`${view.nodes.length} nodes`}>
      <div
        className={
          fullscreen
            ? "fixed inset-4 z-50 rounded-2xl border border-border bg-surface p-4 shadow-card"
            : ""
        }
      >
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge>{nodes.length} raw nodes</Badge>
            <Badge>{edges.length} links</Badge>
            <Badge>{view.groups.length} groups</Badge>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" onClick={() => setCameraZ((value) => Math.max(320, value - 220))}>
              Zoom in
            </Button>
            <Button size="sm" onClick={() => setCameraZ((value) => Math.min(1800, value + 260))}>
              Zoom out
            </Button>
            <Button size="sm" onClick={() => setFitNonce((value) => value + 1)}>
              Fit
            </Button>
            <Button size="sm" onClick={expandAll}>Expand all</Button>
            <Button size="sm" onClick={consolidateAll}>Consolidate</Button>
            <Button size="sm" onClick={() => setFullscreen((value) => !value)}>
              {fullscreen ? "Exit full page" : "Full page"}
            </Button>
          </div>
        </div>
        <div className="grid min-h-0 gap-3 lg:grid-cols-[minmax(0,1fr)_320px]">
          <ThreeGraph
            cameraZ={cameraZ}
            edges={view.edges}
            fitNonce={fitNonce}
            fullscreen={fullscreen}
            nodes={view.nodes}
            onExpand={(node) => setExpandedGroups((previous) => new Set([...previous, node.id]))}
            onSelect={setSelected}
            selectedId={selected?.id}
          />
          <NodeDetails node={selected} />
        </div>
      </div>
    </Panel>
  );
}

function ThreeGraph({
  nodes,
  edges,
  fullscreen,
  cameraZ,
  fitNonce,
  selectedId,
  onSelect,
  onExpand
}: {
  nodes: ViewNode[];
  edges: ViewEdge[];
  fullscreen: boolean;
  cameraZ: number;
  fitNonce: number;
  selectedId?: string;
  onSelect: (node: ViewNode) => void;
  onExpand: (node: ViewNode) => void;
}) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const theme = useUiStore((state) => state.theme);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    mount.innerHTML = "";

    const width = mount.clientWidth || 900;
    const height = mount.clientHeight || 540;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(graphBackground(theme));
    const camera = new THREE.PerspectiveCamera(52, width / height, 1, 5000);
    camera.position.set(0, 0, cameraZ);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setClearColor(graphBackground(theme), 1);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const objects = new Map<string, THREE.Object3D>();
    const positions = seedPositions(nodes);
    const lineMaterial = new THREE.LineBasicMaterial({ color: edgeColor("grouped"), transparent: true, opacity: 0.28 });

    nodes.forEach((node) => {
      const group = createNodeObject(node, selectedId, theme);
      const pos = positions.get(node.id) || { x: 0, y: 0, z: 0 };
      group.position.set(pos.x, pos.y, pos.z);
      group.userData = node;
      objects.set(node.id, group);
      scene.add(group);
    });

    const lines = edges.flatMap((edge) => {
      const source = objects.get(edge.source);
      const target = objects.get(edge.target);
      if (!source || !target) return [];
      const geometry = new THREE.BufferGeometry().setFromPoints([source.position, target.position]);
      const material = lineMaterial.clone();
      material.color = new THREE.Color(edgeColor(edge.kind || ""));
      const line = new THREE.Line(geometry, material);
      scene.add(line);
      return [{ line, edge, geometry }];
    });

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const onClick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const [hit] = raycaster.intersectObjects([...objects.values()], true);
      if (!hit) return;
      const root = findNodeRoot(hit.object);
      const node = root?.userData as ViewNode | undefined;
      if (!node) return;
      onSelect(node);
      if (node.kind === "topic") onExpand(node);
    };
    renderer.domElement.addEventListener("click", onClick);

    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    const onPointerDown = (event: PointerEvent) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return;
      scene.rotation.y += (event.clientX - lastX) * 0.003;
      scene.rotation.x += (event.clientY - lastY) * 0.003;
      lastX = event.clientX;
      lastY = event.clientY;
    };
    const onPointerUp = () => {
      dragging = false;
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      camera.position.z = Math.max(320, Math.min(1800, camera.position.z + event.deltaY * 0.9));
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

    let frame = 0;
    let tick = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      tick += 1;
      if (tick < 260) applyForces(nodes, edges, objects);
      lines.forEach(({ line, edge }) => {
        const source = objects.get(edge.source);
        const target = objects.get(edge.target);
        if (!source || !target) return;
        line.geometry.setFromPoints([source.position, target.position]);
      });
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      renderer.domElement.removeEventListener("click", onClick);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("wheel", onWheel);
      renderer.dispose();
      geometryCleanup(scene);
    };
  }, [cameraZ, edges, fitNonce, fullscreen, nodes, onExpand, onSelect, selectedId, theme]);

  return (
    <div
      className={
        fullscreen
          ? "h-[calc(100vh-125px)] rounded-xl border border-border bg-surface"
          : "h-[min(60vh,540px)] rounded-xl border border-border bg-surface"
      }
      ref={mountRef}
    />
  );
}

function createNodeObject(node: ViewNode, selectedId?: string, theme = "light") {
  const type = node.memory_type || node.kind || "memory";
  const color = colors[type] || colors.memory;
  const group = new THREE.Group();
  const radius = type === "topic" ? 13 : type === "scope" ? 9 : 6.5;
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 24, 16),
    new THREE.MeshBasicMaterial({ color })
  );
  group.add(sphere);
  if (node.id === selectedId) {
    group.add(new THREE.Mesh(
      new THREE.SphereGeometry(radius + 4, 24, 16),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.22 })
    ));
  }

  const label = new SpriteText(node.label || node.id || "node");
  label.color = graphLabelColor(theme);
  label.textHeight = type === "topic" ? 8.5 : 5.8;
  label.position.set(0, radius + 12, 0);
  label.backgroundColor = graphLabelBackground(type, theme);
  label.borderRadius = 4;
  label.padding = type === "topic" ? 4 : 3;
  group.add(label);
  return group;
}

function NodeDetails({ node }: { node: ViewNode | null }) {
  if (!node) {
    return (
      <aside className="rounded-xl border border-border bg-surface p-4 text-sm text-muted">
        Select a node to inspect details. Topic nodes expand on click.
      </aside>
    );
  }
  return (
    <aside className="rounded-xl border border-border bg-surface p-4">
      <div className="font-mono text-xs uppercase tracking-[0.14em] text-muted">
        {node.memory_type || node.kind || "node"}
      </div>
      <h3 className="mt-2 font-display text-lg font-bold text-ink-strong">
        {node.label || node.id}
      </h3>
      <div className="mt-4 grid gap-2 text-sm">
        {node.count && <Detail label="Items" value={`${node.count} consolidated`} />}
        {node.importance != null && (
          <Detail label="Importance" value={`${Math.round(node.importance * 100)}%`} />
        )}
        {node.memory_id && <Detail label="Memory id" value={node.memory_id} />}
        {node.child_ids && <Detail label="Contains" value={`${node.child_ids.length} nodes`} />}
        <Detail label="Node id" value={node.id} />
      </div>
    </aside>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-border pb-2 last:border-b-0">
      <span className="font-mono text-[11px] uppercase tracking-[0.13em] text-muted">
        {label}
      </span>
      <strong className="max-w-[65%] break-words text-right text-ink-strong">{value}</strong>
    </div>
  );
}

function graphViewData(rawNodes: GraphNode[], rawEdges: GraphEdge[], expandedGroups: Set<string>) {
  const groups = new Map<string, GraphNode[]>();
  const passthrough: ViewNode[] = [];
  for (const node of rawNodes) {
    if ((node.memory_type || node.kind) === "scope" || node.kind === "summary") {
      passthrough.push(node);
      continue;
    }
    const label = node.label || "";
    const source = label.match(/^\[web:([^\]]+)\]/)?.[1];
    const key = source ? `Researched ${source}` : `${node.memory_type || node.kind || "memory"}`;
    groups.set(key, [...(groups.get(key) || []), node]);
  }

  const remap = new Map<string, string>();
  const groupedNodes: ViewNode[] = [...passthrough];
  const groupSummaries: ViewNode[] = [];
  passthrough.forEach((node) => remap.set(node.id, node.id));
  groups.forEach((values, key) => {
    const id = `topic:${key}`;
    if (values.length < 4 || expandedGroups.has(id)) {
      values.forEach((node) => {
        remap.set(node.id, node.id);
        groupedNodes.push(node);
      });
      return;
    }
    values.forEach((node) => remap.set(node.id, id));
    const topicNode: ViewNode = {
      id,
      label: `${key} · ${values.length} topics`,
      kind: "topic",
      memory_type: "topic",
      count: values.length,
      child_ids: values.map((node) => node.id)
    };
    groupSummaries.push(topicNode);
    groupedNodes.push(topicNode);
  });

  const edgeMap = new Map<string, ViewEdge>();
  rawEdges.forEach((edge) => {
    const source = remap.get(edge.source);
    const target = remap.get(edge.target);
    if (!source || !target || source === target) return;
    edgeMap.set(`${source}:${target}`, { source, target, kind: "grouped" });
  });

  return { nodes: groupedNodes, edges: [...edgeMap.values()], groups: groupSummaries };
}

function seedPositions(nodes: ViewNode[]) {
  const positions = new Map<string, { x: number; y: number; z: number }>();
  const radius = Math.max(260, nodes.length * 18);
  nodes.forEach((node, index) => {
    const angle = (index / Math.max(1, nodes.length)) * Math.PI * 2;
    const layer = index % 5;
    positions.set(node.id, {
      x: Math.cos(angle) * (radius + layer * 42),
      y: Math.sin(angle) * (radius + layer * 42),
      z: (layer - 2) * 85
    });
  });
  return positions;
}

function applyForces(
  nodes: ViewNode[],
  edges: ViewEdge[],
  objects: Map<string, THREE.Object3D>,
) {
  const nodeObjects = [...objects.values()];
  for (let i = 0; i < nodeObjects.length; i += 1) {
    const a = nodeObjects[i];
    for (let j = i + 1; j < nodeObjects.length; j += 1) {
      const b = nodeObjects[j];
      const diff = new THREE.Vector3().subVectors(b.position, a.position);
      const dist = Math.max(1, diff.length());
      const minDistance = 145;
      if (dist < minDistance) {
        diff.normalize().multiplyScalar((minDistance - dist) * 0.018);
        b.position.add(diff);
        a.position.sub(diff);
      }
    }
  }
  edges.forEach((edge) => {
    const source = objects.get(edge.source);
    const target = objects.get(edge.target);
    if (!source || !target) return;
    const desired = edge.kind === "grouped" ? 360 : 240;
    const diff = new THREE.Vector3().subVectors(target.position, source.position);
    const dist = Math.max(1, diff.length());
    const delta = (dist - desired) * 0.004;
    diff.normalize().multiplyScalar(delta);
    source.position.add(diff);
    target.position.sub(diff);
  });
  nodes.forEach((node) => {
    const object = objects.get(node.id);
    if (!object) return;
    object.position.multiplyScalar(0.998);
  });
}

function findNodeRoot(object: THREE.Object3D | null): THREE.Object3D | null {
  let current = object;
  while (current) {
    if (current.userData && current.userData.id) return current;
    current = current.parent;
  }
  return null;
}

function geometryCleanup(scene: THREE.Scene) {
  scene.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      object.geometry.dispose();
      const material = object.material;
      if (Array.isArray(material)) material.forEach((item) => item.dispose());
      else material.dispose();
    }
  });
}

function graphBackground(theme: string) {
  return theme === "dark" ? "#14120f" : "#f2f1ed";
}

function graphLabelColor(theme: string) {
  return theme === "dark" ? "#fffaf0" : "#26251e";
}

function graphLabelBackground(type: string, theme: string) {
  if (theme === "dark") {
    return type === "topic" ? "rgba(38,37,30,0.88)" : "rgba(38,37,30,0.76)";
  }
  return type === "topic" ? "rgba(247,247,244,0.94)" : "rgba(242,241,237,0.9)";
}

function edgeColor(kind: string) {
  return {
    grouped: "#c08532",
    led_to: "#10b981",
    caused_by: "#ef4444",
    extracted_from: "#6366f1",
    summarizes: "#a855f7"
  }[kind] || "#8f8778";
}
