"""Official INTERACTION Matplotlib map/vehicle helpers with prediction overlays.

The original helper files are unmodified in vehtraj.official (BSD-3-Clause).
This adapter adds challenge case_id selection, model review, and modern controls.
"""
import csv
import json
from pathlib import Path
import numpy as np


class OfficialViewer:
    def __init__(self, csv_path, map_path, case_id='13', review_path=None, sample_index=0):
        from .maps import load_map
        from .explore import validate_review
        from .download import sha256
        from .official.dataset_types import Track, MotionState
        self.csv_path, self.map_path = Path(csv_path), Path(map_path)
        self.map = load_map(self.map_path)
        with self.csv_path.open(newline='', encoding='utf-8-sig') as stream:
            all_rows = list(csv.DictReader(stream))
        self.sample = None
        self.review = json.loads(Path(review_path).read_text()) if review_path else None
        if self.review:
            if self.review['dataset_hashes'].get(self.csv_path.name) != sha256(self.csv_path):
                raise ValueError('Review does not match raw CSV hash')
            validate_review(self.review, all_rows, self.csv_path.name)
            self.sample = next(s for s in self.review['samples'] if s['sample_index'] == sample_index)
            case_id = self.sample['case_id']
        self.case_id = str(case_id)
        self.rows = [r for r in all_rows if r['case_id'] == self.case_id]
        if not self.rows:
            raise ValueError(f'No raw case {case_id}')
        self.frames = sorted({int(r['frame_id']) for r in self.rows})
        self.timestamp = {int(r['frame_id']): int(r['timestamp_ms']) for r in self.rows}
        self.tracks = {}
        for row in self.rows:
            if row['agent_type'] != 'car':
                continue
            tid = row['track_id']
            if tid not in self.tracks:
                track = Track(tid); track.width = float(row['width']); track.length = float(row['length'])
                track.time_stamp_ms_first = int(row['timestamp_ms']); track.time_stamp_ms_last = int(row['timestamp_ms'])
                self.tracks[tid] = track
            track = self.tracks[tid]
            stamp = int(row['timestamp_ms']); state = MotionState(stamp)
            for key in ('x', 'y', 'vx', 'vy', 'psi_rad'):
                setattr(state, key, float(row[key]))
            track.motion_states[stamp] = state
            track.time_stamp_ms_first = min(track.time_stamp_ms_first, stamp)
            track.time_stamp_ms_last = max(track.time_stamp_ms_last, stamp)
        self.patches, self.labels = {}, {}
        self.figure = None; self.index = 0

    def build_figure(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        from .official.map_vis_without_lanelet import draw_map_without_lanelet
        self.figure, self.axes = plt.subplots(figsize=(12, 8))
        self.figure.subplots_adjust(bottom=.19, right=.97, top=.87)
        plt.sca(self.axes)
        draw_map_without_lanelet(str(self.map_path), self.axes, 0.0, 0.0)
        self.axes.set_xlabel('World x (m)'); self.axes.set_ylabel('World y (m)')
        self.figure.suptitle('INTERACTION official renderer | '+(self.review['split']+' predictions' if self.review else 'raw trajectories'), fontsize=15)
        self.title = self.axes.set_title('')
        self.pedestrian = self.axes.scatter([], [], s=18, color='#475569', zorder=24, label='pedestrian/bicycle')
        self.error_line, = self.axes.plot([], [], color='#ce2862', linestyle=':', zorder=34)
        self.predicted_point, = self.axes.plot([], [], 'o', color='#ce2862', zorder=35)
        self.info = self.axes.text(.01, .01, '', transform=self.axes.transAxes, fontsize=9, va='bottom',
                                   bbox={'facecolor':'white','alpha':.88,'edgecolor':'none'}, zorder=40)
        if self.sample:
            s = self.sample
            h, t = np.asarray(s['history_xy']), np.asarray(s['truth_xy'])
            self.axes.plot(h[:,0], h[:,1], color='#1764c0', linewidth=3, label='observed history', zorder=31)
            self.axes.plot(t[:,0], t[:,1], color='#168253', linewidth=3, label='ground-truth future', zorder=31)
            top = int(np.argmax(s['probabilities']))
            for k, points in enumerate(s['prediction_xy']):
                p = np.vstack([h[-1], points]); color = '#ce2862' if k == top else ['#a96413','#834cb7','#ce2862'][k%3]
                self.axes.plot(p[:,0],p[:,1], '--',color=color,linewidth=2.6 if k==top else 1.4,
                               label=f"mode {k+1} p={s['probabilities'][k]:.3f}"+(' (top-1)' if k==top else ''),zorder=32)
            # Include out-of-road predictions instead of clipping away model errors.
            points = np.vstack([h,t,np.asarray(s['prediction_xy']).reshape(-1,2)])
            xlim, ylim = self.axes.get_xlim(), self.axes.get_ylim()
            self.axes.set_xlim(min(xlim[0],points[:,0].min()-3),max(xlim[1],points[:,0].max()+3))
            self.axes.set_ylim(min(ylim[0],points[:,1].min()-3),max(ylim[1],points[:,1].max()+3))
        self.axes.legend(loc='upper left',fontsize=8)
        slider_axes = self.figure.add_axes([.16,.105,.72,.025])
        self.slider = Slider(slider_axes, 'frame', self.frames[0], self.frames[-1], valinit=self.frames[0], valstep=self.frames)
        self.slider.on_changed(lambda value:self.draw_frame(int(value)))
        self.timer = self.figure.canvas.new_timer(interval=100)
        self.timer.add_callback(lambda:self.step(1, loop=True))
        self.buttons = []
        for i, (label, delta) in enumerate([('<<',-10),('<',-1),('play',None),('pause',None),('>',1),('>>',10)]):
            ax = self.figure.add_axes([.2+i*.1,.035,.075,.04]); button = Button(ax,label)
            if label == 'play': button.on_clicked(lambda _:self.timer.start())
            elif label == 'pause': button.on_clicked(lambda _:self.timer.stop())
            else: button.on_clicked(lambda _, d=delta:self.step(d))
            self.buttons.append(button)
        self.draw_frame(self.frames[0])
        return self.figure

    def step(self, delta, loop=False):
        index = self.index + delta
        if loop:index %= len(self.frames)
        index = max(0,min(len(self.frames)-1,index))
        self.slider.set_val(self.frames[index])

    def draw_frame(self, frame):
        from .official.tracks_vis import update_objects_plot
        self.index = self.frames.index(frame); stamp = self.timestamp[frame]
        for patch in self.patches.values():patch.remove()
        for label in self.labels.values():label.remove()
        self.patches.clear();self.labels.clear()
        visible = {k:t for k,t in self.tracks.items() if stamp in t.motion_states}
        update_objects_plot(stamp,self.patches,self.labels,self.axes,track_dict=visible)
        if self.sample and self.sample['track_id'] in self.patches:
            self.patches[self.sample['track_id']].set_color('#1764c0')
        peds = [[float(r['x']),float(r['y'])] for r in self.rows if int(r['frame_id'])==frame and r['agent_type']!='car']
        self.pedestrian.set_offsets(np.asarray(peds).reshape(-1,2))
        self.error_line.set_data([],[]);self.predicted_point.set_data([],[])
        self.title.set_text(f'{self.map_path.stem} | case {self.case_id} | {stamp/1000:.1f} s | frame {frame}')
        if self.sample:
            s=self.sample; m=s['metrics'];top=int(np.argmax(s['probabilities']))
            text=f"{self.review['split']} target {s['track_id']} | top-1 ADE/FDE: {m['top1ADE']:.3f}/{m['top1FDE']:.3f} m | minADE/minFDE: {m['minADE']:.3f}/{m['minFDE']:.3f} m"
            if frame in s['future_frames']:
                j=s['future_frames'].index(frame);p=np.asarray(s['prediction_xy'][top][j]);t=np.asarray(s['truth_xy'][j])
                self.error_line.set_data([p[0],t[0]],[p[1],t[1]]);self.predicted_point.set_data([p[0]],[p[1]])
                text+=f'\nCurrent top-1 displacement: {np.linalg.norm(p-t):.3f} m'
            else:text+='\nObserved history; no future error scored at this frame.'
            self.info.set_text(text)
        else:self.info.set_text('Raw observations only; official Lanelet2 OSM / UTM projection; no model predictions.')
        self.figure.canvas.draw_idle()


def run_viewer(args):
    import matplotlib
    if args.save_png or args.save_gif:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    viewer=OfficialViewer(args.csv,args.map,args.case_id,args.review,args.sample_index)
    fig=viewer.build_figure()
    if args.frame not in viewer.frames:raise ValueError('Requested frame is absent from case')
    viewer.slider.set_val(args.frame)
    if args.save_png:
        path=Path(args.save_png);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=140)
        print(f'official_map_png={path}')
    if args.save_gif:
        from matplotlib.animation import FuncAnimation,PillowWriter
        def update(frame):viewer.slider.set_val(frame)
        animation=FuncAnimation(fig,update,frames=viewer.frames,interval=100,repeat=False)
        path=Path(args.save_gif);path.parent.mkdir(parents=True,exist_ok=True)
        animation.save(path,writer=PillowWriter(fps=10),dpi=100)
        print(f'official_map_gif={path}')
    if not args.save_png and not args.save_gif:plt.show()
    else:plt.close(fig)
