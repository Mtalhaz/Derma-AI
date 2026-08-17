import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class DatabaseHelper {
  static final DatabaseHelper _instance = DatabaseHelper._internal();
  factory DatabaseHelper() => _instance;
  static Database? _database;

  DatabaseHelper._internal();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    String path = join(await getDatabasesPath(), 'lesions_history.db');
    return await openDatabase(
      path,
      version: 1,
      onCreate: _onCreate,
    );
  }

  Future _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE lesions (
        id TEXT PRIMARY KEY,
        image_path TEXT NOT NULL,
        body_part TEXT NOT NULL,
        highest_risk_label TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp INTEGER NOT NULL
      )
    ''');
  }

  Future<int> insertLesion(Map<String, dynamic> lesion) async {
    Database db = await database;
    return await db.insert(
      'lesions',
      lesion,
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<List<Map<String, dynamic>>> getLesions() async {
    Database db = await database;
    return await db.query('lesions', orderBy: 'timestamp DESC');
  }

  Future<int> deleteLesion(String id) async {
    Database db = await database;
    return await db.delete(
      'lesions',
      where: 'id = ?',
      whereArgs: [id],
    );
  }
}
