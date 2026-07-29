extends CanvasLayer
class_name SylvanHUD

# In-game HUD (T0 — signes vitaux) : barres énergie/soif fines + compteurs repas/step, dessinées
# en overlay minimal en haut-gauche par-dessus la vue 3D. VISUAL-ONLY — instancié par main.gd
# UNIQUEMENT sur un vrai écran (jamais headless), comme le décor forêt → zéro impact sur la
# collecte/entraînement. Aucun état de jeu ne vit ici : alimenté chaque frame via update().

const MARGIN := Vector2(16, 14)
const PAD := 12.0
const BAR_W := 168.0
const BAR_H := 11.0
const ROW_H := 22.0
const LABEL_W := 58.0        # colonne gauche pour le libellé de ligne
const NUM_GAP := 10.0        # espace bar → nombre
const NUM_W := 30.0
const FONT_SIZE := 13
const TITLE_SIZE := 12

const COL_PANEL := Color(0.05, 0.06, 0.08, 0.55)
const COL_TEXT := Color(0.92, 0.93, 0.95)
const COL_DIM := Color(0.62, 0.66, 0.72)
const COL_BAR_BG := Color(1, 1, 1, 0.12)
const COL_ENERGY := Color(0.96, 0.74, 0.26)      # ambre = énergie/faim
const COL_LOW := Color(0.90, 0.30, 0.24)         # rouge quand la jauge est critique (<25%)
const COL_THIRST := Color(0.36, 0.66, 0.96)      # bleu = soif (convention eau)

var _canvas: Control
var _font: Font

# État live (posé par update()), fractions 0..1 pour les barres.
var _energy := 1.0
var _thirst := 1.0
var _show_thirst := false
var _meals := 0
var _drinks := 0
var _step := 0
var _max_step := 0
var _terrain := 1.0   # multiplicateur de vitesse SUBI (1 = sol libre, < 1 = sous-bois)
var _ready_ok := false


func _ready() -> void:
	layer = 10  # au-dessus du viewport 3D
	_font = ThemeDB.fallback_font
	_canvas = Control.new()
	_canvas.mouse_filter = Control.MOUSE_FILTER_IGNORE  # ne capte pas la souris
	_canvas.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(_canvas)
	_canvas.draw.connect(_on_draw)
	_ready_ok = true


func update(state: Dictionary) -> void:
	if not _ready_ok:
		return
	_energy = clampf(float(state.get("energy", 1.0)), 0.0, 1.0)
	_thirst = clampf(float(state.get("thirst", 1.0)), 0.0, 1.0)
	_show_thirst = bool(state.get("show_thirst", false))
	_meals = int(state.get("meals", 0))
	_drinks = int(state.get("drinks", 0))
	_step = int(state.get("step", 0))
	_max_step = int(state.get("max_step", 0))
	_terrain = float(state.get("terrain", 1.0))
	_canvas.queue_redraw()


func _bar(pos: Vector2, frac: float, fill: Color) -> void:
	_canvas.draw_rect(Rect2(pos, Vector2(BAR_W, BAR_H)), COL_BAR_BG)
	if frac > 0.0:
		_canvas.draw_rect(Rect2(pos, Vector2(BAR_W * frac, BAR_H)), fill)


func _text(pos: Vector2, s: String, col: Color, size: int = FONT_SIZE) -> void:
	# pos = coin haut-gauche ; draw_string veut une baseline → décalage de l'ascent.
	_canvas.draw_string(_font, pos + Vector2(0, _font.get_ascent(size)), s,
		HORIZONTAL_ALIGNMENT_LEFT, -1, size, col)


func _on_draw() -> void:
	if _font == null:
		return
	var n_bars := 1 + (1 if _show_thirst else 0)

	# Ligne d'état (construite d'abord → dimensionne le panneau).
	var status := "meals %d" % _meals
	if _show_thirst:
		status += "   drinks %d" % _drinks
	status += "   step %d" % _step
	if _max_step > 0:
		status += " / %d" % _max_step
	# SOUS-BOIS CHIFFRÉ. Le ralentissement du terrain est la constante la plus lourde du monde
	# (facteur mesuré 0,635 : budget de trajet 84,9 m -> 53,9 m par vie) et il était INVISIBLE :
	# on ne pouvait ni le voir ni le lire. Les touffes montrent OÙ ; ce nombre montre COMBIEN.
	# Affiché seulement quand il mord, pour que l'apparition du chiffre SOIT l'information.
	if _terrain < 0.995:
		status += "   sous-bois x%.2f" % _terrain

	var bar_block_w := LABEL_W + BAR_W + NUM_GAP + NUM_W
	var status_w: float = _font.get_string_size(status, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE).x
	var content_w: float = maxf(bar_block_w, status_w)
	var panel_w := PAD * 2.0 + content_w
	var panel_h := PAD * 2.0 + 18.0 + n_bars * ROW_H + 18.0

	var origin := MARGIN
	_canvas.draw_rect(Rect2(origin, Vector2(panel_w, panel_h)), COL_PANEL)

	var x := origin.x + PAD
	var y := origin.y + PAD
	_text(Vector2(x, y), "SYLVAN", COL_DIM, TITLE_SIZE)
	y += 18.0

	# Énergie (toujours). Vire au rouge quand critique.
	var e_fill := COL_LOW if _energy < 0.25 else COL_ENERGY
	_text(Vector2(x, y - 1.0), "energy", COL_TEXT)
	_bar(Vector2(x + LABEL_W, y), _energy, e_fill)
	_text(Vector2(x + LABEL_W + BAR_W + NUM_GAP, y - 1.0), str(int(round(_energy * 100.0))), COL_TEXT)
	y += ROW_H

	# Soif (seulement si l'eau existe dans ce monde).
	if _show_thirst:
		var t_fill := COL_LOW if _thirst < 0.25 else COL_THIRST
		_text(Vector2(x, y - 1.0), "thirst", COL_TEXT)
		_bar(Vector2(x + LABEL_W, y), _thirst, t_fill)
		_text(Vector2(x + LABEL_W + BAR_W + NUM_GAP, y - 1.0), str(int(round(_thirst * 100.0))), COL_TEXT)
		y += ROW_H

	_text(Vector2(x, y), status, COL_DIM)
