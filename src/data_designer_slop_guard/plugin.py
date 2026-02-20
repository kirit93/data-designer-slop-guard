from data_designer.plugins.plugin import Plugin, PluginType

slop_guard_plugin = Plugin(
    config_qualified_name="data_designer_slop_guard.config.SlopGuardColumnConfig",
    impl_qualified_name="data_designer_slop_guard.generator.SlopGuardColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
